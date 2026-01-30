import streamlit as st
import pandas as pd
import re
import statsmodels.api as sm
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
import io

st.set_page_config(page_title="Data-Driven Automated Profit Targeting System", layout="wide")

# =============================
# Cache helpers
# =============================
# @st.cache_data
# def load_excel(file) -> pd.ExcelFile:
#     return pd.ExcelFile(file)

# @st.cache_data
# def read_sheet(xls: pd.ExcelFile, sheet_name: str, header=None) -> pd.DataFrame:
#     return pd.read_excel(xls, sheet_name=sheet_name, header=header)

# =============================
# compute_targets_2026 (user-provided)
# =============================

def compute_targets_2026(df_sum, TargetNPATOld, TargetNPATNew, k=None, g=None, gamma=0.8, p=1.2, theta=0.30, tax_rate=0.22, inflation=0.03, elastis_list=None, cap_pct=0.10, small_resid_frac=0.01, outer_max_iter=50):
    df = df_sum.copy(deep=True)
    df['Komponen'] = df['Komponen'].astype(str).str.strip()

    if elastis_list is None:
        elastis_list = ["Cons Fin", "Pend. Adm"]

    for c in ['TargetOld', 'Korelasi_NPAT', 'mi']:
        if c not in df.columns:
            df[c] = np.nan

    if k is None:
        if TargetNPATOld == 0:
            raise ValueError("TargetNPATOld tidak boleh nol untuk menghitung k.")
        k = TargetNPATNew / TargetNPATOld

    if g is None:
        g = (TargetNPATNew - TargetNPATOld) / TargetNPATOld if TargetNPATOld != 0 else 0.0

    df['Elastis'] = df['Komponen'].isin(elastis_list)

    def bounds_from_prev(v):
        v = float(v)
        if v >= 0:
            lower = (1 - cap_pct) * v
            upper = (1 + cap_pct) * v
        else:
            lower = (1 + cap_pct) * v
            upper = (1 - cap_pct) * v
        return lower, upper

    def af_projection(row):
        v = float(row['TargetOld']) if not pd.isna(row['TargetOld']) else 0.0
        c = float(row.get('Korelasi_NPAT', 0.0)) if not pd.isna(row.get('Korelasi_NPAT', np.nan)) else 0.0
        m = float(row.get('mi', 0.0)) if not pd.isna(row.get('mi', np.nan)) else 0.0

        if c == 0 or m == 0 or g == 0:
            v_af = v
        else:
            c_eff_sign = np.sign(c) * (abs(c) ** p)
            if v >= 0:
                factor = 1 + gamma * m * g * c_eff_sign
                v_af = v * factor
            else:
                factor_abs = 1 + gamma * m * g * (abs(c) ** p) * (-np.sign(c))
                v_af = - (abs(v) * factor_abs)

        kom = row['Komponen'].strip().lower()
        if kom in ['biaya kary', 'biaya karyawan', 'biaya_kary', 'biaya kary.']:
            if v_af >= 0:
                v_af = v_af * (1 + inflation)
            else:
                v_af = - (abs(v_af) * (1 + inflation))

        return v_af

    df['AF_pred'] = df.apply(af_projection, axis=1)
    df['k_scale'] = k * df['TargetOld'].astype(float)
    df['Blend'] = (1 - theta) * df['AF_pred'] + theta * df['k_scale']
    df[['Lower','Upper']] = df['TargetOld'].apply(lambda v: pd.Series(bounds_from_prev(v)))

    def cap_val(row):
        x = row['Blend']
        lo, hi = row['Lower'], row['Upper']
        lo_, hi_ = (min(lo,hi), max(lo,hi))
        return min(max(x, lo_), hi_)

    df['Capped'] = df.apply(cap_val, axis=1)
    df['Final'] = df['Capped'].astype(float)

    npat_pred = df['Final'].sum() * (1 - tax_rate)
    residual = TargetNPATNew - npat_pred

    small_thresh = small_resid_frac * abs(TargetNPATNew)
    if abs(residual) <= small_thresh and abs(residual) > 1e-9:
        mask = df['Elastis']
        weight = df.loc[mask, 'AF_pred'].abs()
        if weight.sum() == 0:
            weight = df.loc[mask, 'TargetOld'].abs()
        if weight.sum() > 0:
            adj_nominal_sum = residual / (1 - tax_rate)
            df.loc[mask, 'Final'] += adj_nominal_sum * (weight / weight.sum())
            relax_pct = 0.002
            for idx, r in df.loc[mask].iterrows():
                lo = r['Lower']*(1 - relax_pct) if r['TargetOld']>=0 else r['Lower']*(1 + relax_pct)
                hi = r['Upper']*(1 + relax_pct) if r['TargetOld']>=0 else r['Upper']*(1 - relax_pct)
                df.at[idx, 'Final'] = min(max(df.at[idx, 'Final'], lo), hi)
        residual = TargetNPATNew - df['Final'].sum() * (1 - tax_rate)

    max_iter = 1000
    tol = 1e-6
    it = 0
    while abs(residual) > tol and it < max_iter:
        mask = df['Elastis']
        if residual > 0:
            room = (df.loc[mask, 'Upper'] - df.loc[mask, 'Final']).clip(lower=0)
        else:
            room = (df.loc[mask, 'Final'] - df.loc[mask, 'Lower']).clip(lower=0)

        total_room = room.sum()
        if total_room <= 0:
            break

        cost_mask = (df['TargetOld'] < 0) & mask
        room_adj = room.copy()
        room_adj.loc[cost_mask] *= 0.7

        total_room_adj = room_adj.sum()
        if total_room_adj == 0:
            break

        adj_nominal_sum = residual / (1 - tax_rate)
        alloc = adj_nominal_sum * (room_adj / total_room_adj)

        df.loc[mask, 'Final'] = df.loc[mask, 'Final'] + alloc

        for idx, r in df.loc[mask].iterrows():
            lo, hi = r['Lower'], r['Upper']
            df.at[idx, 'Final'] = min(max(df.at[idx, 'Final'], min(lo,hi)), max(lo,hi))

        residual = TargetNPATNew - df['Final'].sum() * (1 - tax_rate)
        it += 1

    df['TargetNew'] = df['Final'].astype(float)

    # Derived components
    derived_keys = ["Add. Prov WO", "Add. Prov LOR", "Penalty LOR"]

    tol_npat = 1e-3 * max(1.0, abs(TargetNPATNew))
    outer_it = 0
    while outer_it < outer_max_iter:
        def get_val(name):
            sel = df.loc[df['Komponen'].str.strip() == name, 'TargetNew']
            return float(sel.values[0]) if len(sel) > 0 else 0.0

        WO_val = get_val("WO")
        WO_pelsus_val = get_val("WO Pelsus")
        LOR_val = get_val("LOR")

        val_add_prov_wo = 0.25 * (WO_val + WO_pelsus_val)
        val_add_prov_lor = 0.05 * LOR_val
        val_penalty_lor = 0.03 * LOR_val

        for name, value in [("Add. Prov WO", val_add_prov_wo), ("Add. Prov LOR", val_add_prov_lor), ("Penalty LOR", val_penalty_lor)]:
            sel_idx = df.index[df['Komponen'].str.strip() == name].tolist()
            if sel_idx:
                idx0 = sel_idx[0]
                df.at[idx0, 'Final'] = float(value)
                df.at[idx0, 'TargetNew'] = float(value)
            else:
                new_row = {
                    'Komponen': name,
                    'TargetOld': float(value),
                    'Korelasi_NPAT': np.nan,
                    'mi': np.nan,
                    'AF_pred': np.nan,
                    'k_scale': np.nan,
                    'Blend': np.nan,
                    'Lower': bounds_from_prev(value)[0],
                    'Upper': bounds_from_prev(value)[1],
                    'Capped': float(value),
                    'Final': float(value),
                    'TargetNew': float(value),
                    'Elastis': False
                }
                for col in df.columns:
                    if col not in new_row:
                        new_row[col] = np.nan
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        npat_final = df['Final'].sum() * (1 - tax_rate)
        final_diff = TargetNPATNew - npat_final

        if abs(final_diff) <= tol_npat:
            break

        adj_nominal = final_diff / (1 - tax_rate)
        adjustable_mask = df['Elastis'] & (~df['Komponen'].isin(derived_keys))

        if adj_nominal > 0:
            room = (df.loc[adjustable_mask, 'Upper'] - df.loc[adjustable_mask, 'Final']).clip(lower=0.0)
        else:
            room = (df.loc[adjustable_mask, 'Final'] - df.loc[adjustable_mask, 'Lower']).clip(lower=0.0)

        total_room = room.sum()
        if total_room <= 0:
            break

        cost_mask = (df['TargetOld'] < 0) & adjustable_mask
        room_adj = room.copy()
        room_adj.loc[cost_mask.index[cost_mask]] = room_adj.loc[cost_mask.index[cost_mask]] * 0.7 if len(room_adj.loc[cost_mask.index[cost_mask]])>0 else room_adj.loc[cost_mask]

        total_room_adj = room_adj.sum()
        if total_room_adj == 0:
            break

        alloc = adj_nominal * (room_adj / total_room_adj)
        df.loc[adjustable_mask, 'Final'] = df.loc[adjustable_mask, 'Final'] + alloc

        for idx, r in df.loc[adjustable_mask].iterrows():
            lo, hi = r['Lower'], r['Upper']
            df.at[idx, 'Final'] = min(max(df.at[idx, 'Final'], min(lo,hi)), max(lo,hi))
            df.at[idx, 'TargetNew'] = df.at[idx, 'Final']

        outer_it += 1

    npat_final = df['Final'].sum() * (1 - tax_rate)
    final_diff = TargetNPATNew - npat_final
    if abs(final_diff) > 1e-6:
        cand = df[(df['Elastis']) & (df['TargetOld'] > 0) & (~df['Komponen'].isin(derived_keys))]
        if not cand.empty:
            idx_max = cand['AF_pred'].abs().idxmax()
            df.at[idx_max, 'Final'] += final_diff / (1 - tax_rate)
            df.at[idx_max, 'TargetNew'] = df.at[idx_max, 'Final']
            npat_final = df['Final'].sum() * (1 - tax_rate)

    df['TargetNew'] = df['Final'].astype(float)
    npat_final = df['TargetNew'].sum() * (1 - tax_rate)
    if abs(npat_final - TargetNPATNew) > 1e-2 * max(1.0, abs(TargetNPATNew)):
        st.warning(f"Perbedaan akhir NPAT vs Target lebih besar dari toleransi: {npat_final - TargetNPATNew}")

    out_cols = ['Komponen','TargetOld','Korelasi_NPAT','mi','AF_pred','k_scale','Blend','Lower','Upper','Capped','Final','TargetNew','Elastis']
    existing_out_cols = [c for c in out_cols if c in df.columns]
    return df[existing_out_cols].reset_index(drop=True)

# -----------------------------
# BEST MODELS with Iterative Safeguard (MODIFIED per user request, final)
# - removed positive-performer penalty
# - bottom-quartile boost ONLY for negative cost components (WO, WO Pelsus, LOR)
# - Other negative components: NO special weighting (use neutral weights based on S)
# - Regular positive components: NO special weighting (use neutral weights based on S)
# - Add.Prov/Penalty LOR computed from allocated LOR (and WO) then optionally scaled to TargetNew
# - Cons Fin / Pend. Admin favor high-S areas more
# - Recovery Inc favors low-S areas more (worse => more allocation)
# - ensure component sums equal TargetNew (including derived components if TargetNew provided)
# -----------------------------
import numpy as np
import pandas as pd

# -----------------------------
# Parameters (tweakable)
# -----------------------------
WEIGHT_NSA = 0.30
WEIGHT_TAF = 0.30
WEIGHT_ACH = 0.40

GAMMA = 0.1             # ± relative change bound (10%)
EPS = 1e-9
MIN_ADJUST = 0.0         # minimum adjustment to avoid 0% growth
MAX_ITER_REDISTR = 500   # safeguard for redistrib loop
TOL = 1e-9               # tolerance residual

# New tuning params (tweak as needed)
NEG_BOTTOM_BOOST = 2.0   # bottom-quartile boost intensity for special negative components
CONS_PEND_POWER = 2.0
RECOVERY_BAD_WEIGHT = 2.0

# Names (exact strings expected in df_growth Komponen)
CMP_WO = "WO"
CMP_WO_PELSUS = "WO Pelsus"
CMP_LOR = "LOR"
CMP_ADD_PROV_WO = "Add. Prov WO"
CMP_ADD_PROV_LOR = "Add. Prov LOR"
CMP_PENALTY_LOR = "Penalty LOR"
CMP_CONS_FIN = "Cons Fin"
CMP_PEND_ADMIN = "Pend. Admin"
CMP_RECOVERY = "Recovery Inc."

# -----------------------------
# Helpers
# -----------------------------
def pct_rank(s):
    return s.rank(method='average', pct=True).fillna(0.5)

def compute_score(df_area, w_nsa=WEIGHT_NSA, w_taf=WEIGHT_TAF, w_ach=WEIGHT_ACH):
    # expects df_area to have columns: 'NSA','TARGET AF','AchNPAT(%)'
    s_nsa = pct_rank(df_area['NSA'])
    s_taf = pct_rank(df_area['TARGET AF'])
    s_ach = pct_rank(df_area['AchNPAT(%)'])
    S = w_nsa * s_nsa + w_taf * s_taf + w_ach * s_ach
    S = (S - S.min()) / max(EPS, (S.max() - S.min()))
    return S

def build_weights_for_component(component_name, S_arr, ach_arr):
    """
    Return non-negative weights (not normalized) of same length as S_arr.

    Behavior:
      - Cons Fin / Pend. Admin: accentuate good S by larger power
      - Recovery Inc.: favors poor performing areas (1 - S), amplified by RECOVERY_BAD_WEIGHT and bottom quartile factor
      - Negative special components (WO, WO Pelsus, LOR): base weight (0.5+0.5*S) but boosted on bottom-quartile by NEG_BOTTOM_BOOST proportional to distance to Q1
      - ALL OTHER components (positive or negative): use S-based neutral weights (base = 0.5 + 0.5*S)
        => this replaces the previous equal-equal weighting and ensures allocation follows area scoring from NSA/AF/Ach.
    """
    S_arr = np.asarray(S_arr, dtype=float)
    ach_arr = np.asarray(ach_arr, dtype=float)
    n = len(S_arr)

    # bottom fraction (distance to Q1 normalized) computed from AchNPAT for use where needed
    q25 = np.quantile(ach_arr, 0.25)
    denom = max(EPS, q25 - ach_arr.min())
    bottom_frac = np.zeros(n)
    for i in range(n):
        if ach_arr[i] <= q25:
            bottom_frac[i] = (q25 - ach_arr[i]) / denom

    # Special behavior retained for listed components
    if component_name in (CMP_CONS_FIN, CMP_PEND_ADMIN):
        # Favor high-S areas more strongly
        base = 0.5 + 0.5 * S_arr  # in [0.5,1]
        w = base ** CONS_PEND_POWER
    elif component_name == CMP_RECOVERY:
        # Favor low-S (worse performing) areas, amplified by bottom_frac
        w = (0.5 + 0.5 * (1.0 - S_arr))
        w = w * (1.0 + RECOVERY_BAD_WEIGHT * bottom_frac)
    elif component_name in (CMP_WO, CMP_WO_PELSUS, CMP_LOR):
        # Special negative components: base weighted by S but bottom-quartile boost applied
        base = 0.5 + 0.5 * S_arr
        w = base * (1.0 + NEG_BOTTOM_BOOST * bottom_frac)
    else:
        # **UPDATED**: All other components -> use S-based neutral weights (not flat ones)
        # This makes allocation proportional to area score S (derived from NSA, TARGET AF, AchNPAT),
        # with base in [0.5, 1.0] so no area gets zero weight.
        w = 0.5 + 0.5 * S_arr

    w = np.maximum(w, EPS)
    return w

def sign_safe_sum_equalize(x_signed, target_signed):
    """
    Helper: if x_signed is an array whose sum is nearly target_signed, return x_signed
    otherwise scale magnitudes so that sum(x_signed) == target_signed while preserving sign pattern.
    Works for both positive and negative targets.
    """
    s = x_signed.sum()
    tgt = float(target_signed)
    if abs(s - tgt) < 1e-6:
        return x_signed
    # If both negative or both positive scale magnitudes
    if (s < 0 and tgt < 0) or (s > 0 and tgt > 0):
        scale = tgt / (s if abs(s) > EPS else EPS)
        return x_signed * scale
    # If signs differ (rare), scale magnitudes of absolute values to match target sign
    mags = np.abs(x_signed)
    if tgt < 0:
        mags_sum = mags.sum()
        if mags_sum < EPS: 
            return np.full_like(x_signed, tgt / len(x_signed))
        scaled = - (mags / mags_sum) * abs(tgt)
        return scaled
    else:
        mags_sum = mags.sum()
        if mags_sum < EPS:
            return np.full_like(x_signed, tgt / len(x_signed))
        scaled = (mags / mags_sum) * tgt
        return scaled

# -----------------------------
# Core allocator (sign-aware) for a single component
# -----------------------------
def allocate_component(a0, T2026, S_arr, ach_arr, component_name,
                       gamma=GAMMA):
    """
    a0: array of historical per-area amounts (signed)
    T2026: scalar target total for component (signed)
    S_arr: score array (0..1)
    ach_arr: AchNPAT(%) array (raw % values)
    component_name: string
    returns allocated per-area array (signed), diagnostics
    """
    a0 = np.asarray(a0, dtype=float)
    n = len(a0)
    # determine sign (component treated as negative if target negative OR historical all negative)
    is_neg = (float(T2026) < 0) or (np.all(a0 < 0))

    if is_neg:
        # operate on magnitudes then sign negative at end
        m0 = np.abs(a0).astype(float)
        L_mag = m0 * (1.0 - gamma)
        U_mag = m0 * (1.0 + gamma)
        m_x = m0.copy()

        w = build_weights_for_component(component_name, S_arr, ach_arr)

        target_mag = abs(float(T2026))
        it = 0
        while it < MAX_ITER_REDISTR:
            residual = target_mag - m_x.sum()
            if abs(residual) <= TOL:
                break
            if residual > 0:
                room = np.maximum(U_mag - m_x, 0.0)
                if room.sum() < EPS:
                    break
                alloc = w / max(EPS, w.sum()) * residual
                alloc = np.minimum(alloc, room)
                m_x += alloc
            else:
                reducible = np.maximum(m_x - L_mag, 0.0)
                need_cut = -residual
                if reducible.sum() < EPS:
                    break
                cuts = w / max(EPS, w.sum()) * need_cut
                cuts = np.minimum(cuts, reducible)
                m_x -= cuts
            it += 1

        # clip
        m_x = np.minimum(np.maximum(m_x, L_mag), U_mag)
        # scale final magnitudes to ensure exact total = target_mag
        sum_mx = m_x.sum()
        if sum_mx < EPS:
            # fallback: distribute equally
            m_x = np.full_like(m_x, target_mag / max(1, n))
        else:
            m_x = m_x * (target_mag / sum_mx)

        x_signed = -m_x
        diagnostics = {'iterations': it, 'final_residual_mag': float(target_mag - m_x.sum())}
        # ensure exact equality (numerical)
        x_signed = sign_safe_sum_equalize(x_signed, -target_mag)
        return x_signed, diagnostics

    else:
        # POSITIVE component - allocate on signed space
        L = a0 * (1.0 - gamma)
        U = a0 * (1.0 + gamma)
        x = a0.copy().astype(float)

        w = build_weights_for_component(component_name, S_arr, ach_arr)

        it = 0
        while it < MAX_ITER_REDISTR:
            residual = float(T2026) - x.sum()
            if abs(residual) <= TOL:
                break
            if residual > 0:
                room = np.maximum(U - x, 0.0)
                if room.sum() < EPS:
                    break
                alloc = w / max(EPS, w.sum()) * residual
                alloc = np.minimum(alloc, room)
                x += alloc
            else:
                reducible = np.maximum(x - L, 0.0)
                need_cut = -residual
                if reducible.sum() < EPS:
                    break
                cuts = w / max(EPS, w.sum()) * need_cut
                cuts = np.minimum(cuts, reducible)
                x -= cuts
            it += 1

        # clip
        x = np.minimum(np.maximum(x, L), U)
        # scale final to exact target
        sum_x = x.sum()
        if abs(sum_x) < EPS:
            x = np.full_like(x, float(T2026) / max(1, n))
        else:
            x = x * (float(T2026) / sum_x)

        diagnostics = {'iterations': it, 'final_residual': float(T2026 - x.sum())}
        x = sign_safe_sum_equalize(x, float(T2026))
        return x, diagnostics

# -----------------------------
# Top-level function
# -----------------------------
def allocate_npat(df_growth, df_pivotArea, df_pivotHistTargetArea,
                  gamma=GAMMA):
    """
    df_growth: dataframe with columns ['Komponen','TargetOld','TargetNew']
    df_pivotArea: area-level current indicators, must contain columns ['Area','NSA','TARGET AF','AchNPAT(%)']
    df_pivotHistTargetArea: historical per-area component values (columns include area and component names; 'WILAYAH' accepted)
    Returns:
      A_alloc: DataFrame [Area x Komponen] allocated values
      df_growth_area: growth % per area per component (sign-aware)
      diagnostics: dict of diagnostics per component
    Notes:
      - Add. Prov WO, Add. Prov LOR, Penalty LOR computed from allocated WO/WO Pelsus/LOR
      - If df_growth has TargetNew for those derived components, we scale the computed values to match that total
    """
    # normalize input area column name from hist table
    df_area = df_pivotArea.copy()
    df_hist = df_pivotHistTargetArea.copy()
    if 'WILAYAH' in df_hist.columns and 'Area' not in df_hist.columns:
        df_hist = df_hist.rename(columns={'WILAYAH': 'Area'})

    # find common areas
    common = [a for a in df_area['Area'].astype(str).tolist()
              if a in df_hist['Area'].astype(str).tolist()]
    if len(common) == 0:
        raise ValueError("No matching Area keys between df_pivotArea and df_pivotHistTargetArea")

    df_area = df_area.set_index('Area').loc[common].reset_index()
    df_hist = df_hist.set_index('Area').loc[common].reset_index()

    # compute S and arrays
    S = compute_score(df_area)
    df_area['S'] = S
    S_arr = df_area.set_index('Area')['S'].reindex(common).values
    ach_arr = df_area.set_index('Area')['AchNPAT(%)'].reindex(common).values

    components = df_growth['Komponen'].tolist()
    components = list(dict.fromkeys(components))  # preserve order, unique

    # Build historical A0 (if missing, zeros)
    A0 = pd.DataFrame(index=common, columns=components, dtype=float)
    for c in components:
        if c in df_hist.columns:
            A0[c] = df_hist[c].astype(float).values
        else:
            A0[c] = np.zeros(len(common), dtype=float)

    A_alloc = pd.DataFrame(index=common, columns=components, dtype=float)
    diagnostics = {}

    # Components to skip initially (derived ones)
    skip_components = {CMP_ADD_PROV_WO, CMP_ADD_PROV_LOR, CMP_PENALTY_LOR}
    main_components = [c for c in components if c not in skip_components]

    # Ensure WO, WO Pelsus, LOR allocated first
    priority = [CMP_WO, CMP_WO_PELSUS, CMP_LOR]
    ordered_main = []
    for p in priority:
        if p in main_components:
            ordered_main.append(p)
    for c in main_components:
        if c not in ordered_main:
            ordered_main.append(c)

    # Allocate main components, matching TargetNew exactly per component
    for c in ordered_main:
        row = df_growth.loc[df_growth['Komponen'] == c]
        if row.empty:
            continue
        T2025 = float(row['TargetOld'].values[0])
        T2026 = float(row['TargetNew'].values[0])
        a0 = A0[c].fillna(0.0).astype(float).values

        x, diag = allocate_component(a0, T2026, S_arr, ach_arr, c, gamma=gamma)
        A_alloc[c] = x
        diagnostics[c] = {'T2025': T2025, 'T2026': T2026, **diag}

    # Compute derived components from allocated WO/WO Pelsus/LOR
    # Formulas:
    #   Add. Prov WO = 25%*(WO + WO Pelsus)
    #   Add. Prov LOR = 5%*(LOR)
    #   Penalty LOR = 5%*(LOR)
    # After computing, if df_growth contains TargetNew for these derived components,
    # scale the computed per-area results so their sum equals TargetNew.
    def compute_and_optionally_scale(name, arr_values, df_growth):
        vals = np.array(arr_values).astype(float)
        # find target in df_growth if exists
        row = df_growth.loc[df_growth['Komponen'] == name]
        if not row.empty:
            tgt = float(row['TargetNew'].values[0])
            sum_vals = vals.sum()
            if abs(sum_vals) < EPS:
                # distribute target proportionally to historical A0 sums if possible, else equally
                if name in A0.columns and A0[name].abs().sum() > EPS:
                    base = np.abs(A0[name].astype(float).values)
                    denom = base.sum()
                    if denom < EPS:
                        scaled = np.full_like(vals, tgt / max(1, len(vals)))
                    else:
                        scaled = (base / denom) * tgt
                else:
                    scaled = np.full_like(vals, tgt / max(1, len(vals)))
                vals = scaled
            else:
                vals = vals * (tgt / sum_vals)
        return vals

    # Prepare inputs
    n = len(common)
    # safe get columns (if missing use zeros)
    wo_arr = A_alloc[ CMP_WO ].astype(float).values if CMP_WO in A_alloc.columns else np.zeros(n)
    wo_pelsus_arr = A_alloc[ CMP_WO_PELSUS ].astype(float).values if CMP_WO_PELSUS in A_alloc.columns else np.zeros(n)
    lor_arr = A_alloc[ CMP_LOR ].astype(float).values if CMP_LOR in A_alloc.columns else np.zeros(n)

    # compute derived raw
    addprovwo_raw = 0.25 * (wo_arr + wo_pelsus_arr)
    addprovlor_raw = 0.05 * lor_arr
    penalylor_raw = 0.05 * lor_arr

    # optionally scale to df_growth targets if provided
    if CMP_ADD_PROV_WO in components:
        vals = compute_and_optionally_scale(CMP_ADD_PROV_WO, addprovwo_raw, df_growth)
        A_alloc[CMP_ADD_PROV_WO] = vals
        diagnostics[CMP_ADD_PROV_WO] = {'sum_computed': float(addprovwo_raw.sum()), 'sum_final': float(vals.sum())}

    if CMP_ADD_PROV_LOR in components:
        vals = compute_and_optionally_scale(CMP_ADD_PROV_LOR, addprovlor_raw, df_growth)
        A_alloc[CMP_ADD_PROV_LOR] = vals
        diagnostics[CMP_ADD_PROV_LOR] = {'sum_computed': float(addprovlor_raw.sum()), 'sum_final': float(vals.sum())}

    if CMP_PENALTY_LOR in components:
        vals = compute_and_optionally_scale(CMP_PENALTY_LOR, penalylor_raw, df_growth)
        A_alloc[CMP_PENALTY_LOR] = vals
        diagnostics[CMP_PENALTY_LOR] = {'sum_computed': float(penalylor_raw.sum()), 'sum_final': float(vals.sum())}

    # Final check: ensure every component sum equals df_growth TargetNew (if component present in df_growth)
    for c in components:
        row = df_growth.loc[df_growth['Komponen'] == c]
        if row.empty:
            continue
        tgt = float(row['TargetNew'].values[0])
        if c not in A_alloc.columns:
            # if not computed, fill with zeros then scale to tgt evenly
            A_alloc[c] = np.zeros(n, dtype=float)
            if abs(tgt) > EPS:
                A_alloc[c] = np.full(n, tgt / max(1, n))
            diagnostics[c] = diagnostics.get(c, {})
            diagnostics[c].update({'final_adjusted_sum': float(A_alloc[c].sum())})
        else:
            sum_c = float(A_alloc[c].sum())
            if abs(sum_c - tgt) > 1e-6:
                # scale preserving sign pattern
                arr = A_alloc[c].astype(float).values
                arr = sign_safe_sum_equalize(arr, tgt)
                A_alloc[c] = arr
                diagnostics[c] = diagnostics.get(c, {})
                diagnostics[c].update({'final_adjusted_sum': float(arr.sum()), 'was_scaled_to_target': True})
            else:
                diagnostics[c] = diagnostics.get(c, {})
                diagnostics[c].update({'final_adjusted_sum': sum_c, 'was_scaled_to_target': False})

    # Compute growth percentages (sign-aware) per area per component
    df_growth_area = pd.DataFrame(index=common, columns=components, dtype=float)
    for c in components:
        a0 = A0[c].astype(float).values
        anew = A_alloc[c].astype(float).values if c in A_alloc.columns else np.zeros(len(common), dtype=float)
        row = df_growth.loc[df_growth['Komponen'] == c]
        if row.empty:
            continue
        is_neg = float(row['TargetOld'].values[0]) < 0
        if is_neg:
            mag0 = np.abs(a0)
            magn = np.abs(anew)
            denom = mag0.copy()
            denom[denom < EPS] = np.maximum(EPS, magn[denom < EPS])
            growth = (magn - mag0) / denom * 100.0
        else:
            denom = np.abs(a0)
            denom[denom < EPS] = np.maximum(EPS, np.abs(anew[denom < EPS]))
            growth = (anew - a0) / denom * 100.0
        df_growth_area[c] = growth

    A_alloc.index.name = 'Area'
    df_growth_area.index.name = 'Area'
    return A_alloc, df_growth_area, diagnostics

# -----------------------------
# Example usage:
# -----------------------------

# -----------------------------
# Area -> Cabang allocator (full final version)
# Mechanism matched to the main allocator rules exactly
# -----------------------------
import numpy as np
import pandas as pd

# -----------------------------
# Parameters (tweakable)
# -----------------------------
WEIGHT_NSA = 0.30
WEIGHT_TAF = 0.30
WEIGHT_ACH = 0.40

GAMMA = 0.08            # ± relative change bound (10%)
EPS = 1e-9
MAX_ITER_REDISTR = 500
TOL = 1e-9

NEG_BOTTOM_BOOST = 2.0
CONS_PEND_POWER = 2.0
RECOVERY_BAD_WEIGHT = 2.0

# component name constants (must match names in your df_alloc / df_growth)
CMP_WO = "WO"
CMP_WO_PELSUS = "WO Pelsus"
CMP_LOR = "LOR"
CMP_ADD_PROV_WO = "Add. Prov WO"
CMP_ADD_PROV_LOR = "Add. Prov LOR"
CMP_PENALTY_LOR = "Penalty LOR"
CMP_CONS_FIN = "Cons Fin"
CMP_PEND_ADMIN = "Pend. Admin"
CMP_RECOVERY = "Recovery Inc."

# -----------------------------
# Helpers (same logic as area allocator)
# -----------------------------
def pct_rank(s):
    return s.rank(method='average', pct=True).fillna(0.5)

def compute_score(df_area_like, w_nsa=WEIGHT_NSA, w_taf=WEIGHT_TAF, w_ach=WEIGHT_ACH):
    """
    Compute S in [0,1] using 'NSA','TARGET AF','AchNPAT(%)' for the given dataframe (Area-like or Cabang-level).
    If AchNPAT(%) missing attempt to compute from NPAT MTD / TARGET MTD.
    """
    df = df_area_like.copy()
    if 'AchNPAT(%)' not in df.columns:
        if ('NPAT MTD' in df.columns) and ('TARGET MTD' in df.columns):
            denom = df['TARGET MTD'].replace(0, np.nan)
            df['AchNPAT(%)'] = (df['NPAT MTD'] / denom).fillna(0.0)
        else:
            df['AchNPAT(%)'] = 0.0
    s_nsa = pct_rank(df['NSA'])
    s_taf = pct_rank(df['TARGET AF'])
    s_ach = pct_rank(df['AchNPAT(%)'])
    S = w_nsa * s_nsa + w_taf * s_taf + w_ach * s_ach
    S = (S - S.min()) / max(EPS, (S.max() - S.min()))
    return S

def build_weights_for_component(component_name, S_arr, ach_arr):
    """
    Return non-negative weights array corresponding to S_arr.
    Exactly matches behavior described:
      - Cons Fin / Pend. Admin: base^(CONS_PEND_POWER)
      - Recovery: favors low S (1-S) amplified by bottom quartile factor
      - WO / WO Pelsus / LOR: base=(0.5+0.5*S) multiplied by bottom-quartile boost (NEG_BOTTOM_BOOST * bottom_frac)
      - All other components: S-based neutral weights = 0.5 + 0.5*S
    """
    S_arr = np.asarray(S_arr, dtype=float)
    ach_arr = np.asarray(ach_arr, dtype=float)
    n = len(S_arr)

    q25 = np.quantile(ach_arr, 0.25)
    denom = max(EPS, q25 - ach_arr.min())
    bottom_frac = np.zeros(n)
    for i in range(n):
        if ach_arr[i] <= q25:
            bottom_frac[i] = (q25 - ach_arr[i]) / denom

    if component_name in (CMP_CONS_FIN, CMP_PEND_ADMIN):
        base = 0.5 + 0.5 * S_arr
        w = base ** CONS_PEND_POWER
    elif component_name == CMP_RECOVERY:
        w = (0.5 + 0.5 * (1.0 - S_arr))
        w = w * (1.0 + RECOVERY_BAD_WEIGHT * bottom_frac)
    elif component_name in (CMP_WO, CMP_WO_PELSUS, CMP_LOR):
        base = 0.5 + 0.5 * S_arr
        w = base * (1.0 + NEG_BOTTOM_BOOST * bottom_frac)
    else:
        # non-special components: neutral S-based weights (not flat)
        w = 0.5 + 0.5 * S_arr

    w = np.maximum(w, EPS)
    return w

def sign_safe_sum_equalize(x_signed, target_signed):
    """
    Preserve sign pattern while making sum(x_signed) == target_signed.
    """
    s = x_signed.sum()
    tgt = float(target_signed)
    if abs(s - tgt) < 1e-6:
        return x_signed
    if (s < 0 and tgt < 0) or (s > 0 and tgt > 0):
        scale = tgt / (s if abs(s) > EPS else EPS)
        return x_signed * scale
    mags = np.abs(x_signed)
    if tgt < 0:
        mags_sum = mags.sum()
        if mags_sum < EPS:
            return np.full_like(x_signed, tgt / len(x_signed))
        scaled = - (mags / mags_sum) * abs(tgt)
        return scaled
    else:
        mags_sum = mags.sum()
        if mags_sum < EPS:
            return np.full_like(x_signed, tgt / len(x_signed))
        scaled = (mags / mags_sum) * tgt
        return scaled

# allocate_component copied exactly (sign-aware iterative redistrib + clipping + final scaling)
def allocate_component(a0, T2026, S_arr, ach_arr, component_name, gamma=GAMMA):
    a0 = np.asarray(a0, dtype=float)
    n = len(a0)
    is_neg = (float(T2026) < 0) or (np.all(a0 < 0))

    if is_neg:
        m0 = np.abs(a0).astype(float)
        L_mag = m0 * (1.0 - gamma)
        U_mag = m0 * (1.0 + gamma)
        m_x = m0.copy()

        w = build_weights_for_component(component_name, S_arr, ach_arr)

        target_mag = abs(float(T2026))
        it = 0
        while it < MAX_ITER_REDISTR:
            residual = target_mag - m_x.sum()
            if abs(residual) <= TOL:
                break
            if residual > 0:
                room = np.maximum(U_mag - m_x, 0.0)
                if room.sum() < EPS:
                    break
                alloc = w / max(EPS, w.sum()) * residual
                alloc = np.minimum(alloc, room)
                m_x += alloc
            else:
                reducible = np.maximum(m_x - L_mag, 0.0)
                need_cut = -residual
                if reducible.sum() < EPS:
                    break
                cuts = w / max(EPS, w.sum()) * need_cut
                cuts = np.minimum(cuts, reducible)
                m_x -= cuts
            it += 1

        m_x = np.minimum(np.maximum(m_x, L_mag), U_mag)
        sum_mx = m_x.sum()
        if sum_mx < EPS:
            m_x = np.full_like(m_x, target_mag / max(1, n))
        else:
            m_x = m_x * (target_mag / sum_mx)

        x_signed = -m_x
        diagnostics = {'iterations': it, 'final_residual_mag': float(target_mag - m_x.sum())}
        x_signed = sign_safe_sum_equalize(x_signed, -target_mag)
        return x_signed, diagnostics

    else:
        L = a0 * (1.0 - gamma)
        U = a0 * (1.0 + gamma)
        x = a0.copy().astype(float)

        w = build_weights_for_component(component_name, S_arr, ach_arr)

        it = 0
        while it < MAX_ITER_REDISTR:
            residual = float(T2026) - x.sum()
            if abs(residual) <= TOL:
                break
            if residual > 0:
                room = np.maximum(U - x, 0.0)
                if room.sum() < EPS:
                    break
                alloc = w / max(EPS, w.sum()) * residual
                alloc = np.minimum(alloc, room)
                x += alloc
            else:
                reducible = np.maximum(x - L, 0.0)
                need_cut = -residual
                if reducible.sum() < EPS:
                    break
                cuts = w / max(EPS, w.sum()) * need_cut
                cuts = np.minimum(cuts, reducible)
                x -= cuts
            it += 1

        x = np.minimum(np.maximum(x, L), U)
        sum_x = x.sum()
        if abs(sum_x) < EPS:
            x = np.full_like(x, float(T2026) / max(1, n))
        else:
            x = x * (float(T2026) / sum_x)

        diagnostics = {'iterations': it, 'final_residual': float(T2026 - x.sum())}
        x = sign_safe_sum_equalize(x, float(T2026))
        return x, diagnostics

# -----------------------------
# allocate_area_to_cabang (final)
# -----------------------------
def allocate_area_to_cabang(df_alloc_area, df_pivotCabang, df_pivotHistTargetCabang=None, df_growth_area=None, gamma=GAMMA):
    """
    Map df_alloc_area (Area x Komponen) down to Cabang using the same allocation mechanics.
    Returns: A_alloc_cabang (MultiIndex Area,Cabang), df_growth_cabang (growth % vs hist), diagnostics dict.
    """
    # normalize df_alloc_area
    df_alloc = df_alloc_area.copy()
    if 'Area' in df_alloc.columns:
        df_alloc = df_alloc.set_index('Area')

    # prepare df_pivotCabang
    df_cab = df_pivotCabang.copy()
    for col in ['NPAT MTD','TARGET MTD','NSA','TARGET AF']:
        if col in df_cab.columns:
            df_cab[col] = pd.to_numeric(df_cab[col], errors='coerce').fillna(0.0)
        else:
            df_cab[col] = 0.0

    if 'AchNPAT(%)' not in df_cab.columns:
        denom = df_cab['TARGET MTD'].replace(0, np.nan)
        df_cab['AchNPAT(%)'] = (df_cab['NPAT MTD'] / denom).fillna(0.0)

    # historical per-cabang (if provided)
    if df_pivotHistTargetCabang is None:
        df_hist_cab = pd.DataFrame(columns=['Area','Cabang'])
    else:
        df_hist_cab = df_pivotHistTargetCabang.copy()
        if 'WILAYAH' in df_hist_cab.columns and 'Area' not in df_hist_cab.columns:
            df_hist_cab = df_hist_cab.rename(columns={'WILAYAH':'Area'})

    components = list(df_alloc.columns)

    # build MultiIndex of (Area,Cabang) ensuring areas present in df_alloc are included
    areas = df_alloc.index.astype(str).tolist()
    rows = []
    for area in areas:
        sub = df_cab[df_cab['Area'].astype(str) == str(area)]
        if sub.shape[0] == 0:
            rows.append((area, str(area)))
        else:
            for c in sub['Cabang'].astype(str).tolist():
                rows.append((area, c))
    index = pd.MultiIndex.from_tuples(rows, names=['Area','Cabang'])
    A_alloc_cabang = pd.DataFrame(0.0, index=index, columns=components, dtype=float)
    diagnostics = {}

    # build historical A0_cab (indexed by MultiIndex)
    A0_cab_df = pd.DataFrame(0.0, index=index, columns=components, dtype=float)
    if not df_hist_cab.empty:
        df_hist_cab = df_hist_cab.copy()
        if 'Cabang' not in df_hist_cab.columns and 'ID CAB' in df_hist_cab.columns:
            df_hist_cab = df_hist_cab.rename(columns={'ID CAB':'Cabang'})
        df_hist_cab['Area'] = df_hist_cab['Area'].astype(str)
        df_hist_cab['Cabang'] = df_hist_cab['Cabang'].astype(str)
        df_hist_cab_indexed = df_hist_cab.set_index(['Area','Cabang'], drop=False)
        for comp in components:
            vals = []
            for a,c in index:
                try:
                    v = float(df_hist_cab_indexed.loc[(a,c), comp]) if (a,c) in df_hist_cab_indexed.index else 0.0
                except Exception:
                    v = 0.0
                vals.append(v)
            A0_cab_df[comp] = np.array(vals, dtype=float)

    # group cabang by area for quick access
    df_cab_group = { area: df_cab[df_cab['Area'].astype(str)==str(area)].copy() for area in areas }

    # allocate component-by-component, area-by-area
    for comp in components:
        diagnostics[comp] = {}
        for area in areas:
            area_target = float(df_alloc.loc[area, comp]) if area in df_alloc.index else 0.0
            sub = df_cab_group.get(area, pd.DataFrame())
            if sub.shape[0] == 0:
                # synthetic single cabang under area
                cab_names = [str(area)]
                S_arr = np.array([0.0])
                ach_arr = np.array([0.0])
                a0 = np.array([ float(A0_cab_df.loc[(area,str(area)), comp]) ]) if (area,str(area)) in A0_cab_df.index else np.array([0.0])
                alloc_vals = np.array([area_target], dtype=float)
                diag = {'iterations':0, 'note':'synthetic single-cabang'}
            else:
                cab_names = sub['Cabang'].astype(str).tolist()
                df_sub = sub.copy()
                if 'AchNPAT(%)' not in df_sub.columns:
                    denom = df_sub['TARGET MTD'].replace(0, np.nan)
                    df_sub['AchNPAT(%)'] = (df_sub['NPAT MTD'] / denom).fillna(0.0)
                S_series = compute_score(df_sub)
                S_arr = S_series.values
                ach_arr = df_sub['AchNPAT(%)'].values

                a0_list = []
                for c in cab_names:
                    a0_list.append(float(A0_cab_df.loc[(area,c), comp]) if (area,c) in A0_cab_df.index else 0.0)
                a0 = np.array(a0_list, dtype=float)

                alloc_vals, diag = allocate_component(a0, area_target, S_arr, ach_arr, comp, gamma=gamma)

            # write allocations
            for i, c in enumerate(cab_names):
                A_alloc_cabang.loc[(area, c), comp] = float(alloc_vals[i])

            diagnostics[comp][area] = {**diag, 'n_cabang': len(cab_names), 'area_target': area_target}

    # Derived components (computed from allocated WO/LOR)
    for nm in [CMP_WO, CMP_WO_PELSUS, CMP_LOR]:
        if nm not in A_alloc_cabang.columns:
            A_alloc_cabang[nm] = 0.0

    wo_vals = A_alloc_cabang[CMP_WO]
    wo_pelsus_vals = A_alloc_cabang[CMP_WO_PELSUS] if CMP_WO_PELSUS in A_alloc_cabang.columns else pd.Series(0.0, index=A_alloc_cabang.index)
    lor_vals = A_alloc_cabang[CMP_LOR]

    addprovwo_raw = 0.25 * (wo_vals + wo_pelsus_vals)
    addprovlor_raw = 0.05 * lor_vals
    penalylor_raw = 0.05 * lor_vals

    def compute_and_scale_derived_per_area(name, raw_vals_per_index):
        result = raw_vals_per_index.copy()
        # if df_alloc (area targets) has this derived component -> scale per-area to match area target
        if name in df_alloc.columns:
            for area in areas:
                # find indices for that area
                idx_area = [idx for idx in result.index if idx[0] == area]
                if len(idx_area) == 0:
                    continue
                raw_area_vals = result.loc[idx_area].astype(float).values
                sum_raw = raw_area_vals.sum() if raw_area_vals.size > 0 else 0.0
                tgt_area = float(df_alloc.loc[area, name]) if area in df_alloc.index else None
                if tgt_area is None:
                    continue
                if abs(sum_raw) < EPS:
                    # fallback to historical proportions if available else equal split
                    if name in A0_cab_df.columns and A0_cab_df.loc[idx_area].abs().sum() > EPS:
                        base = np.abs(A0_cab_df.loc[idx_area, name].astype(float).values)
                        denom = base.sum()
                        if denom < EPS:
                            scaled = np.full(len(idx_area), tgt_area / max(1, len(idx_area)))
                        else:
                            scaled = (base / denom) * tgt_area
                    else:
                        scaled = np.full(len(idx_area), tgt_area / max(1, len(idx_area)))
                    result.loc[idx_area] = scaled
                else:
                    factor = tgt_area / sum_raw if sum_raw != 0 else 1.0
                    result.loc[idx_area] = raw_area_vals * factor
        return result

    addprovwo_scaled = compute_and_scale_derived_per_area(CMP_ADD_PROV_WO, addprovwo_raw)
    addprovlor_scaled = compute_and_scale_derived_per_area(CMP_ADD_PROV_LOR, addprovlor_raw)
    penalylor_scaled = compute_and_scale_derived_per_area(CMP_PENALTY_LOR, penalylor_raw)

    A_alloc_cabang[CMP_ADD_PROV_WO] = addprovwo_scaled
    A_alloc_cabang[CMP_ADD_PROV_LOR] = addprovlor_scaled
    A_alloc_cabang[CMP_PENALTY_LOR] = penalylor_scaled

    # Final safety scaling to ensure sums per-area equal area-targets (preserve sign pattern)
    for comp in components:
        if comp not in df_alloc.columns:
            continue
        for area in areas:
            area_target = float(df_alloc.loc[area, comp])
            idx_area = [idx for idx in A_alloc_cabang.index if idx[0] == area]
            if len(idx_area) == 0:
                continue
            arr = A_alloc_cabang.loc[idx_area, comp].astype(float).values
            s = arr.sum()
            if abs(s - area_target) > 1e-6:
                scaled = sign_safe_sum_equalize(arr, area_target)
                A_alloc_cabang.loc[idx_area, comp] = scaled
                diagnostics.setdefault(comp, {}).setdefault('final_scaling', {})[area] = {'before_sum': float(s), 'after_sum': float(scaled.sum())}

    df_growth_cabang = pd.DataFrame(index=A_alloc_cabang.index, columns=components, dtype=float)

    for comp in components:
        # --- Tentukan apakah komponen dianggap negatif atau positif ---
        if df_growth_area is not None and "Komponen" in df_growth_area.columns:
            row = df_growth_area.loc[df_growth_area["Komponen"] == comp]
            if not row.empty:
                is_neg = float(row["TargetNew"].values[0]) < 0
            else:
                # fallback: cek mayoritas / magnitude di histori
                if comp in A0_cab_df.columns:
                    vals = A0_cab_df[comp].dropna().values
                    if len(vals) > 0:
                        frac_neg = (vals < 0).sum() / len(vals)
                        neg_mag = abs(vals[vals < 0]).sum()
                        pos_mag = abs(vals[vals > 0]).sum()
                        is_neg = frac_neg >= 0.5 or neg_mag > pos_mag
                    else:
                        is_neg = False
                else:
                    is_neg = False
        else:
            if comp in A0_cab_df.columns:
                vals = A0_cab_df[comp].dropna().values
                if len(vals) > 0:
                    frac_neg = (vals < 0).sum() / len(vals)
                    neg_mag = abs(vals[vals < 0]).sum()
                    pos_mag = abs(vals[vals > 0]).sum()
                    is_neg = frac_neg >= 0.5 or neg_mag > pos_mag
                else:
                    is_neg = False
            else:
                is_neg = False

        # --- Loop per cabang ---
        for idx in A_alloc_cabang.index:
            a0_val = float(A0_cab_df.loc[idx, comp]) if comp in A0_cab_df.columns else 0.0
            anew_val = float(A_alloc_cabang.loc[idx, comp]) if comp in A_alloc_cabang.columns else 0.0

            if is_neg:
                # basis magnitude (untuk komponen negatif, misalnya biaya/kerugian)
                mag0 = abs(a0_val)
                magn = abs(anew_val)
                denom = mag0 if mag0 >= EPS else max(EPS, magn)
                growth = (magn - mag0) / denom * 100.0
            else:
                # basis langsung (untuk komponen positif)
                denom = abs(a0_val) if abs(a0_val) >= EPS else max(EPS, abs(anew_val))
                growth = (anew_val - a0_val) / denom * 100.0

            df_growth_cabang.loc[idx, comp] = growth

    # Pastikan index multi-level (Area, Cabang) diberi nama
    A_alloc_cabang.index.names = ["Area", "Cabang"]
    df_growth_cabang.index.names = ["Area", "Cabang"]


    return A_alloc_cabang, df_growth_cabang, diagnostics


# -----------------------------
# Example usage:
# -----------------------------
# df_alloc  = DataFrame indexed by Area, columns = Komponen (result of allocate_npat)
# df_pivotCabang = DataFrame with columns ['Area','Cabang','NPAT MTD','TARGET MTD','NSA','TARGET AF', ...]
# df_pivotHistTargetCabang optional contains historical per-cabang per-komponen columns (Area,Cabang,...)
# df_growth_area optional = df_growth used at Area-level (to detect negative components via TargetOld)
#

# -----------------------------

# --- definisikan bulan jatuh Idul Fitri per tahun (isi sesuai data resmi) ---
idulfitri_months = {
    2016: [7],   # Juli 2016
    2017: [6],   # Juni 2017
    2018: [6],   # Juni 2018
    2019: [6],   # Juni 2019
    2020: [5],   # Mei 2020
    2021: [5],   # Mei 2021
    2022: [5],   # Mei 2022
    2023: [4],   # April 2023
    2024: [4],   # April 2024
    2025: [3],   # Maret 2025
}

def days_in_month(year, month):
    return pd.Period(f"{year}-{month:02d}").days_in_month

# fungsi untuk flag idul fitri
def is_idulfitri(year, month):
    return 1 if month in idulfitri_months.get(year, []) else 0

# Natal selalu di bulan 12
def is_natal(month):
    return 1 if month == 12 else 0

# Tahun baru selalu di bulan 1
def is_tahunbaru(month):
    return 1 if month == 1 else 0

# Bulan Pendek  selalu di bulan 2
def is_bulanpendek(month):
    return 1 if month == 2 else 0

# =========================
# 1) Persiapan data
# =========================
def prepare_df(df):
    df = df.copy()
    df['BULAN'] = df['BULAN'].astype(str)
    df['date'] = pd.to_datetime(df['BULAN'], format='%Y%m')
    df['year'] = df['Tahun'].astype(int)
    df['month'] = df['Bulan_Num'].astype(int)

    # Hitung total NPAT per tahun
    yearly_total = df.groupby('year')["Pendapatan PK Gross"].transform("sum")
    df['prop_npat'] = df['Pendapatan PK Gross'] / yearly_total

    return df

# =========================
# 2) Train dengan XGBoost (proporsi)
# =========================
def train_xgb_prop(df, train_years, valid_years):
    feature_cols = ['month','Jumlah Hari','IdulFitri','Natal','TahunBaru','BulanPendek']
    
    df_train = df[df['year'].isin(train_years)]
    df_valid = df[df['year'].isin(valid_years)]

    X_train, y_train = df_train[feature_cols], df_train['prop_npat']
    X_valid, y_valid = df_valid[feature_cols], df_valid['prop_npat']

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 3,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42
    }

    evals = [(dtrain, "train"), (dvalid, "valid")]
    model = xgb.train(params, dtrain, num_boost_round=500,
                      evals=evals, early_stopping_rounds=30, verbose_eval=50)

    # Prediksi train & valid
    df_train['pred'] = model.predict(dtrain)
    df_valid['pred'] = model.predict(dvalid)

    # =========================
    # Evaluasi
    # =========================
    def mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    mae_train = mean_absolute_error(df_train['prop_npat'], df_train['pred'])
    mae_valid = mean_absolute_error(df_valid['prop_npat'], df_valid['pred'])
    r2_train = r2_score(df_train['prop_npat'], df_train['pred'])
    r2_valid = r2_score(df_valid['prop_npat'], df_valid['pred'])
    mape_train = mape(df_train['prop_npat'], df_train['pred'])
    mape_valid = mape(df_valid['prop_npat'], df_valid['pred'])

    print("=== Evaluasi Model ===")
    print(f"Train MAE: {mae_train:.4f}, R²: {r2_train:.4f}, MAPE: {mape_train:.2f}%")
    print(f"Valid MAE: {mae_valid:.4f}, R²: {r2_valid:.4f}, MAPE: {mape_valid:.2f}%")

    # Cek sum per tahun (apakah ≈ 1)
    print("\nProporsi per tahun (valid):")
    print(df_valid.groupby("year")[["prop_npat","pred"]].sum())

    return model, feature_cols, df_train, df_valid


def predict_prop(year_target, model, feature_cols, idul_month=None):
    months = range(1,13)
    days_in_month = [31,28,31,30,31,30,31,31,30,31,30,31]
    # cek leap year
    if (year_target % 4 == 0 and year_target % 100 != 0) or (year_target % 400 == 0):
        days_in_month[1] = 29

    df_pred = pd.DataFrame({
        "year": year_target,
        "month": months,
        "Jumlah Hari": days_in_month,
        "Natal": [1 if m==12 else 0 for m in months],
        "TahunBaru": [1 if m==1 else 0 for m in months],
        "BulanPendek": [1 if m==2 else 0 for m in months],
        "IdulFitri": [1 if (idul_month is not None and m==idul_month) else 0 for m in months]
    })

    dmat = xgb.DMatrix(df_pred[feature_cols])
    df_pred["pred_prop"] = model.predict(dmat)

    # Normalisasi agar sum = 1
    df_pred["pred_prop"] = df_pred["pred_prop"] / df_pred["pred_prop"].sum()

    return df_pred[["year","month","Jumlah Hari","IdulFitri","Natal","TahunBaru","BulanPendek","pred_prop"]]

def get_contour(year_target, model, feature_cols, idul_month=None):
    df_pred = predict_prop(year_target, model, feature_cols, idul_month=idul_month)

    # Normalisasi ulang agar total = 1 (sudah ada di predict_prop, tapi untuk aman)
    df_pred["pred_prop"] = df_pred["pred_prop"] / df_pred["pred_prop"].sum()

    # Hitung contour (avg = 1)
    mean_prop = df_pred["pred_prop"].mean()
    df_pred["contour"] = df_pred["pred_prop"] / mean_prop

    return df_pred[["year","month","pred_prop","contour"]]

# =========================
# 1) Persiapan data
# =========================
def prepare_df2(df):
    df = df.copy()
    df['BULAN'] = df['BULAN'].astype(str)
    df['date'] = pd.to_datetime(df['BULAN'], format='%Y%m')
    df['year'] = df['Tahun'].astype(int)
    df['month'] = df['Bulan_Num'].astype(int)

    return df

# =========================
# 2) Train dengan XGBoost (proporsi)
# =========================
def train_xgb_prop2(df, train_years, valid_years):
    feature_cols = ['month','Jumlah Hari','IdulFitri','Natal','TahunBaru','BulanPendek']
    
    df_train = df[df['year'].isin(train_years)]
    df_valid = df[df['year'].isin(valid_years)]

    X_train, y_train = df_train[feature_cols], df_train['coll']
    X_valid, y_valid = df_valid[feature_cols], df_valid['coll']

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 3,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42
    }

    evals = [(dtrain, "train"), (dvalid, "valid")]
    model = xgb.train(params, dtrain, num_boost_round=500,
                      evals=evals, early_stopping_rounds=30, verbose_eval=50)

    # Prediksi train & valid
    df_train['pred'] = model.predict(dtrain)
    df_valid['pred'] = model.predict(dvalid)

    # =========================
    # Evaluasi
    # =========================
    def mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    mae_train = mean_absolute_error(df_train['coll'], df_train['pred'])
    mae_valid = mean_absolute_error(df_valid['coll'], df_valid['pred'])
    r2_train = r2_score(df_train['coll'], df_train['pred'])
    r2_valid = r2_score(df_valid['coll'], df_valid['pred'])
    mape_train = mape(df_train['coll'], df_train['pred'])
    mape_valid = mape(df_valid['coll'], df_valid['pred'])

    print("=== Evaluasi Model ===")
    print(f"Train MAE: {mae_train:.4f}, R²: {r2_train:.4f}, MAPE: {mape_train:.2f}%")
    print(f"Valid MAE: {mae_valid:.4f}, R²: {r2_valid:.4f}, MAPE: {mape_valid:.2f}%")

    # Cek sum per tahun (apakah ≈ 1)
    print("\nProporsi per tahun (valid):")
    print(df_valid.groupby("year")[["coll","pred"]].sum())

    return model, feature_cols, df_train, df_valid

# =========================
# 3) Prediksi untuk tahun target
# =========================

def predict_prop2(year_target, model, feature_cols, idul_month=None):
    months = range(1,13)
    days_in_month = [31,28,31,30,31,30,31,31,30,31,30,31]
    # cek leap year
    if (year_target % 4 == 0 and year_target % 100 != 0) or (year_target % 400 == 0):
        days_in_month[1] = 29

    df_pred = pd.DataFrame({
        "year": year_target,
        "month": months,
        "Jumlah Hari": days_in_month,
        "Natal": [1 if m==12 else 0 for m in months],
        "TahunBaru": [1 if m==1 else 0 for m in months],
        "BulanPendek": [1 if m==2 else 0 for m in months],
        "IdulFitri": [1 if (idul_month is not None and m==idul_month) else 0 for m in months]
    })

    dmat = xgb.DMatrix(df_pred[feature_cols])
    df_pred["pred_coll"] = model.predict(dmat)

    # Normalisasi agar sum = 1
    df_pred["pred_coll"] = df_pred["pred_coll"] / df_pred["pred_coll"].sum()

    return df_pred[["year","month","Jumlah Hari","IdulFitri","Natal","TahunBaru","BulanPendek","pred_coll"]]

def midpoint_contour(df1, df2):
    # gabungkan berdasarkan month
    merged = pd.merge(df1, df2, on="month", suffixes=("_1", "_2"))
    # rata-rata contour
    merged["contour_mid"] = (merged["contour_1"] + merged["contour_2"]) / 2
    # normalisasi supaya total = 12
    merged["contour_mid"] = merged["contour_mid"] * (12 / merged["contour_mid"].sum())

    return merged[["month", "contour_mid"]]

# =======================
# 1. NASIONAL BULANAN
# =======================
def hitung_nasional_bulanan(A_alloc_nasional, contour_mid):
    contour = contour_mid[["month", "contour_mid"]].copy().set_index("month")
    
    nasional_fy = A_alloc_nasional.copy()
    nasional_bulanan = (
        nasional_fy
        .assign(dummy=1)
        .merge(contour.reset_index(), how="cross")  # cross join → 12 bulan
    )
    nasional_bulanan["TargetBulanan"] = (
        (nasional_bulanan["TargetNew"] / 12) * nasional_bulanan["contour_mid"]
    )
    return nasional_fy, nasional_bulanan

# =======================
# 2. AREA BULANAN
# =======================
def hitung_area_bulanan(A_alloc_area, contour_mid):
    contour = contour_mid[["month", "contour_mid"]].copy().set_index("month")
    
    area_fy = A_alloc_area.copy()
    area_bulanan = (
        area_fy.reset_index().assign(dummy=1)
        .merge(contour.reset_index(), how="cross")
    )

    for col in A_alloc_area.columns:
        area_bulanan[col] = (area_bulanan[col] / 12) * area_bulanan["contour_mid"]

    area_bulanan = area_bulanan.set_index(["Area", "month"])
    return area_bulanan

# =======================
# 3. CABANG BULANAN
# =======================
def hitung_cabang_bulanan(A_alloc_cabang, contour_mid):
    contour = contour_mid[["month", "contour_mid"]].copy().set_index("month")
    
    cabang_fy = A_alloc_cabang.copy()
    cabang_bulanan = (
        cabang_fy.reset_index().assign(dummy=1)
        .merge(contour.reset_index(), how="cross")
    )

    for col in A_alloc_cabang.columns:
        cabang_bulanan[col] = (cabang_bulanan[col] / 12) * cabang_bulanan["contour_mid"]

    cabang_bulanan = cabang_bulanan.set_index(["Area", "Cabang", "month"])
    return cabang_bulanan

# =======================
# 4. WRAPPER EXPORT
# =======================
def allocate_with_contour(A_alloc_nasional, A_alloc_area, A_alloc_cabang, contour_mid, output_file="alokasi.xlsx"):
    # hitung masing-masing
    nasional_fy, nasional_bulanan = hitung_nasional_bulanan(A_alloc_nasional, contour_mid)
    area_fy, area_bulanan = hitung_area_bulanan(A_alloc_area, contour_mid)
    cabang_fy, cabang_bulanan = hitung_cabang_bulanan(A_alloc_cabang, contour_mid)

    # export ke Excel
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        nasional_fy.to_excel(writer, sheet_name="Nasional Full Year", index=False)
        nasional_bulanan.to_excel(writer, sheet_name="Nasional Bulanan", index=False)

        area_fy.to_excel(writer, sheet_name="Area Full Year")
        area_bulanan.to_excel(writer, sheet_name="Area Bulanan")

        cabang_fy.to_excel(writer, sheet_name="Cabang Full Year")
        cabang_bulanan.to_excel(writer, sheet_name="Cabang Bulanan")

    return {
        "nasional_fy": nasional_fy,
        "nasional_bulanan": nasional_bulanan,
        "area_fy": area_fy,
        "area_bulanan": area_bulanan,
        "cabang_fy": cabang_fy,
        "cabang_bulanan": cabang_bulanan,
    }


def get_contour2(year_target, model, feature_cols, idul_month=None):
    df_pred = predict_prop2(year_target, model, feature_cols, idul_month=idul_month)

    # Normalisasi ulang agar total = 1 (sudah ada di predict_prop, tapi untuk aman)
    df_pred["pred_coll"] = df_pred["pred_coll"] / df_pred["pred_coll"].sum()

    # Hitung contour (avg = 1)
    mean_coll = df_pred["pred_coll"].mean()
    df_pred["contour"] = df_pred["pred_coll"] / mean_coll

    return df_pred[["year","month","pred_coll","contour"]]



# ============================
# SumoPod AI Integration
# ============================

import json
from openai import OpenAI


def df_to_json_payload(A_alloc_area: pd.DataFrame):
    """
    Convert DataFrame A_alloc_area menjadi payload JSON yang siap dianalisis AI
    """
    df = A_alloc_area.copy().reset_index()

    payload = {
        "metadata": {
            "source": "Automated Profit Targeting System",
            "level": "Area",
            "period": "2026",
            "description": "Hasil alokasi target per area dan komponen"
        },
        "data": df.to_dict(orient="records")
    }
    return payload


def call_sumopod_analysis(payload_json):
    """
    Kirim data ke SumoPod dan minta analisis + rekomendasi strategis
    """
    client = OpenAI(
        api_key="sk-oOwA4SWEReeYqQVmE3tkxQ",
        base_url="https://ai.sumopod.com/v1"
    )

    prompt = f"""
Anda adalah konsultan bisnis, keuangan, dan risk management senior
yang terbiasa menganalisis Branch Profitability (BP) dan Profit Planning (PP)
di industri pembiayaan / consumer finance.

⚠️ KONTEKS PENTING (WAJIB DIPAHAMI):
Data yang dianalisis adalah RANCANGAN TARGET cabang (budgeting & planning),
bukan realisasi historis.
• Nilai NEGATIF bukan berarti kerugian aktual
• Nilai negatif merepresentasikan KOMPONEN PENGURANG PROFIT
  (cost, risk cost, provisioning, atau penalty)
• Fokus analisis adalah KUALITAS PERENCANAAN TARGET, bukan performa aktual

=====================================
DEFINISI & LOGIKA BISNIS KOMPONEN
=====================================

Pendapatan (Revenue Driver):
- Cons Fin            : Target pendapatan bunga pembiayaan konsumen
- Pend. Adm           : Target pendapatan administrasi kontrak
- Pend. Denda         : Target pendapatan denda keterlambatan
- Pend. PT&PP Others  : Pendapatan pelunasan dipercepat & payment point
- Recovery Inc.       : Target pemulihan dari kontrak WO
- Pend. Asuransi Siaga: Pendapatan perlindungan risiko pembiayaan
- Pend. Lain-lain     : Pendapatan non-operasional pendukung

Biaya Operasional & Akuisisi:
- COF                 : Cost of Fund sebagai biaya pendanaan
- Biaya Mktng         : Biaya promosi & insentif akuisisi
- Biaya Mktng_2       : Biaya akibat splitting rate / customer behavior
- Biaya Kary          : Biaya SDM cabang
- Bbn Adm. Umum       : Biaya operasional & administrasi cabang

Komponen Risiko & Loss Planning:
- WO                  : Target write off kontrak bermasalah
- WO Pelsus           : Target loss akibat pelunasan khusus
- Add. Prov WO        : Tambahan pencadangan risiko WO
- LOR                 : Target kerugian dari recovery non-flag WO
- Add. Prov LOR       : Tambahan pencadangan risiko LOR
- Penalty LOR         : Penalti akibat target recovery tidak tercapai
- Biaya Denda+Rec+LOR : Biaya penagihan, recovery, dan handling risiko

=====================================
TUJUAN ANALISIS
=====================================

Berdasarkan data alokasi TARGET per AREA dan KOMPONEN berikut, lakukan:

1️⃣ Analisis Pola Target
   - Pola distribusi revenue, cost, dan risk cost antar area
   - Keseimbangan antara agresivitas target dan kontrol risiko
   - Identifikasi struktur target yang terlalu optimistis atau terlalu konservatif

2️⃣ Identifikasi Area Risiko & Opportunity
   - Area dengan beban risiko (WO, LOR, provisioning) relatif tinggi
   - Area dengan potensi leverage pendapatan (Cons Fin, Recovery, Denda)
   - Area dengan struktur biaya tidak proporsional terhadap revenue

3️⃣ Insight Strategis (Tajam & Kontekstual)
   - Apa implikasi target ini terhadap strategi operasional cabang?
   - Risiko tersembunyi yang mungkin muncul jika target dijalankan apa adanya
   - Trade-off antara growth, profitability, dan risk appetite

4️⃣ Rekomendasi Aksi Bisnis (KONKRET & PRIORITAS)
   - Rekomendasi perbaikan struktur target (rebalancing)
   - Area mana yang perlu:
     • dikontrol risikonya
     • didorong pertumbuhannya
     • dioptimalkan efisiensinya
   - Rekomendasi harus realistis, actionable, dan berorientasi eksekusi cabang

Gunakan sudut pandang manajemen senior.
Berikan insight yang tajam, berbasis logika bisnis,
bukan sekadar deskripsi angka.

=====================================
DATA TARGET:
=====================================
{json.dumps(payload_json, indent=2)}
"""


    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.6,
        max_tokens=800
    )

    return response.choices[0].message.content


# =============================
# Streamlit UI
# =============================
st.title("Data-Driven Profit Target Recommendation System")

# =====================================
# 🎯 Pilih Mode Perhitungan
# =====================================
tab1, tab2 = st.tabs(["🧮 Full Pipeline (dari Nasional)", "🏁 Mulai dari Target Area FY"])

# ============================
# TAB 1: FULL PIPELINE (existing)
# ============================
with tab1:
    # seluruh isi app lama (dari uploaded_file = st.file_uploader ... sampai akhir)
    # tidak perlu diubah — cukup letakkan semua blok lama di dalam `with tab1:`
    uploaded_file = st.file_uploader("Pilih file Excel (.xls/.xlsx)", type=["xls", "xlsx"], key="main_file")

    if uploaded_file is None:
        st.info("Silakan upload file Excel untuk memulai.")
    else:
        try:
            xls = pd.ExcelFile(uploaded_file)
            sheets = xls.sheet_names
            col1, col2 = st.columns(2)
            with col1:
                sheet_df = st.selectbox("Pilih sheet untuk: Komponen Profit Y-1", sheets, index=0)
            with col2:
                sheet_hist = st.selectbox("Pilih sheet untuk: Data Historis", sheets, index=min(1, len(sheets)-1))

            df = pd.read_excel(uploaded_file, sheet_name=sheet_df)
            df_full_hist = pd.read_excel(uploaded_file, sheet_name=sheet_hist, header=3)

            st.subheader("Preview: Target Komponen Y-1")
            st.dataframe(df.head())
            st.subheader("Preview: Data Historis")
            st.dataframe(df_full_hist.head())

            # ⬇️ Semua kode lama kamu mulai dari sini
            df_hist2 = df_full_hist.copy()
            df_histTarget = df.copy()

            # year selection
            year_source = None
            if 'Tahun' in df_full_hist.columns:
                years = sorted(df_full_hist['Tahun'].dropna().unique().astype(str).tolist())
                year_source = 'Tahun'
            else:
                candidate = None
                for c in ['BULAN', 'Periode', 'Bulan']:
                    if c in df_full_hist.columns:
                        candidate = c
                        break
                if candidate is not None:
                    yrs = df_full_hist[candidate].dropna().astype(str).apply(lambda x: x[:4] if len(str(x))>=4 else x)
                    years = sorted(yrs.unique().tolist())
                    year_source = candidate
                else:
                    years = []

            if len(years) == 0:
                st.warning("Tidak ditemukan kolom tahun/BULAN/Periode pada df_full_hist. Anda tetap bisa lanjut tetapi filter tahun tidak tersedia.")
                selected_years = []
            else:
                selected_years = st.multiselect("Pilih historis tahun yang akan digunakan (multi-select)", years, default=years)

            TargetNPATNew = st.number_input('Masukkan Target NPAT Tahun Target', value=5020026762188.356, format="%f")

            run_label = "Jalankan Perhitungan Alokasi Target"
            if st.button(run_label):
                try:
                    # ------------------
                    # NASIONAL
                    # ------------------
                    df_filtered = df[df["Periode"].notna() & (df["Periode"] != "Grand Total")] if 'Periode' in df.columns else df.copy()
                    df_sum = df_filtered.select_dtypes(include=[np.number]).sum()
                    df_sum = df_sum.reset_index()
                    df_sum.columns = ["Komponen", "TargetOld"]
                    drop_list = ["Branch", "Gross Profit", "Opr Profit", "Pro NSA 2025", "NPBT", "Tax", "TARGET NPAT 2025", "TAX"]
                    df_sum = df_sum[~df_sum["Komponen"].isin(drop_list)]
                    df_sum["Komponen"] = df_sum["Komponen"].astype(str).str.strip()

                    korelasi_map = {
                        "Cons Fin": 0.831130389432443,
                        "Titipan": -0.715897973334781,
                        "Pend. Adm": 0.783700896812182,
                        "COF": -0.76338413185167,
                        "Biaya Mktng": -0.853592715994968,
                        "Biaya Mktng_2": -0.817674204508504,
                        "Biaya Kary": -0.3429084324886,
                        "Bbn Adm. Umum": -0.641181915841652,
                        "Pend. Denda": 0.673318192592633,
                        "Pend. PT&PP - Others": -0.259403169581807,
                        "Recovery Inc.": 0.259983964622261,
                        "Pend. Lain-lain": -0.261034369148943,
                        "Pend. Asuransi Siaga": 0.00289391129544453,
                        "WO": -0.715302140202459,
                        "WO Pelsus": -0.640828405160015,
                        "Add. Prov WO": -0.421535597200164,
                        "LOR": -0.798774333441749,
                        "Add. Prov LOR": 0.00213619764559999,
                        "Penalty LOR": -0.373944635105175,
                        "Biaya Denda+Rec+Lor": -0.528624485103896
                    }

                    df_sum["Korelasi_NPAT"] = df_sum["Komponen"].map(korelasi_map)

                    df_full_hist.columns = df_full_hist.columns.str.strip()

                    cols_keep = [
                        "BULAN","TARGET AF","TARGET MTD","Pendapatan PK Gross","Titipan","Pendapatan Administrasi",
                        "By Bank CBG (+COF)","Biaya Marketing","Biaya Mktng_2","Biaya Karyawan","Beban Adm umum dicatat di Cbg",
                        "Bbn Adm Umum dicatat di HO","Penalty Document Cbg","Pendapatan denda","Penalty Overbook",
                        "Pendapatan Pre Termination","Pendptan ex AR Write Off","Pendapatan Lain-Lain","Pendapatan Asuransi Siaga",
                        "WO AR","WO Pelsus","Additional Provision AR","Rugi penjualan  AYD Realized (data UTJ)",
                        "Rugi penjualan  AYD Realized (tambahan data GL)","Additional Provision AYD Unrealized","Penalty LOR",
                        "Biaya-Biaya (Denda+Rec+Lor)","NPAT MTD"
                    ]

                    cols_keep_exist = [c for c in cols_keep if c in df_full_hist.columns]
                    if len(cols_keep_exist) == 0:
                        st.error("Kolom historis yang dibutuhkan tidak ditemukan. Periksa format sheet historis.")
                        st.stop()

                    df_hist = df_full_hist[cols_keep_exist].copy()

                    if selected_years and year_source is not None:
                        if year_source == 'Tahun' and 'Tahun' in df_full_hist.columns:
                            df_hist = df_hist[df_full_hist['Tahun'].astype(str).isin(selected_years)].copy()
                        else:
                            col_candidate = year_source
                            yrs_series = df_hist[col_candidate].astype(str).apply(lambda x: x[:4] if len(x)>=4 else x)
                            df_hist = df_hist[yrs_series.isin(selected_years)].copy()

                    df_hist = df_hist.dropna()

                    if 'Beban Adm umum dicatat di Cbg' in df_hist.columns and 'Bbn Adm Umum dicatat di HO' in df_hist.columns:
                        df_hist['Bbn Adm'] = df_hist['Beban Adm umum dicatat di Cbg'] + df_hist['Bbn Adm Umum dicatat di HO']
                    if all(c in df_hist.columns for c in ['Penalty Document Cbg','Pendapatan denda','Penalty Overbook']):
                        df_hist['Pendapatan Denda'] = df_hist['Penalty Document Cbg'] + df_hist['Pendapatan denda'] + df_hist['Penalty Overbook']
                    if all(c in df_hist.columns for c in ['Rugi penjualan  AYD Realized (data UTJ)','Rugi penjualan  AYD Realized (tambahan data GL)']):
                        df_hist['LOR'] = df_hist['Rugi penjualan  AYD Realized (data UTJ)'] + df_hist['Rugi penjualan  AYD Realized (tambahan data GL)']

                    drop_candidates = [
                        'Beban Adm umum dicatat di Cbg','Bbn Adm Umum dicatat di HO','Penalty Document Cbg','Pendapatan denda',
                        'Penalty Overbook','Rugi penjualan  AYD Realized (data UTJ)','Rugi penjualan  AYD Realized (tambahan data GL)'
                    ]
                    drop_candidates = [c for c in drop_candidates if c in df_hist.columns]
                    if drop_candidates:
                        df_hist = df_hist.drop(columns=drop_candidates)

                    if 'BULAN' in df_hist.columns:
                        df_hist = df_hist.groupby('BULAN', as_index=False).sum(numeric_only=True)

                    cols = df_hist.columns.tolist()
                    cols = [col.replace('Sum of ', '').strip() for col in cols]
                    mapping = {
                        'BULAN': 'Periode','TARGET MTD': 'Target NPAT','Pendapatan PK Gross': 'Cons Fin','Titipan': 'Titipan',
                        'Pendapatan Administrasi': 'Pend. Adm','By Bank CBG (+COF)': 'COF','Biaya Marketing': 'Biaya Mktng',
                        'Biaya Mktng_2': 'Biaya Mktng_2','Biaya Karyawan': 'Biaya Kary','Bbn Adm': 'Bbn Adm. Umum',
                        'Pendapatan Denda': 'Pend. Denda','Pendapatan Pre Termination': 'Pend. PT&PP - Others',
                        'Pendptan ex AR Write Off': 'Recovery Inc.','Pendapatan Lain-Lain': 'Pend. Lain-lain',
                        'Pendapatan Asuransi Siaga': 'Pend. Asuransi Siaga','WO AR': 'WO','WO Pelsus': 'WO Pelsus',
                        'Additional Provision AR': 'Add. Prov WO','LOR': 'LOR','Additional Provision AYD Unrealized': 'Add. Prov LOR',
                        'Penalty LOR': 'Penalty LOR','Biaya-Biaya (Denda+Rec+Lor)': 'Biaya Denda+Rec+Lor'
                    }
                    cols_mapped = [mapping.get(col, col) for col in cols]
                    df_hist.columns = cols_mapped

                    if 'Periode' in df_hist.columns:
                        df_hist['Periode'] = df_hist['Periode'].astype(str)
                        df_hist = df_hist[~df_hist['Periode'].str.startswith(('2017','2018','2019','2020','2021'))].copy()

                    df_sum['Komponen'] = df_sum['Komponen'].astype(str).str.strip()
                    komponen_cols = [c for c in df_hist.columns if c not in ['Periode','TARGET AF','Target NPAT']]

                    results = []
                    for col in komponen_cols:
                        try:
                            y = df_hist[col].astype(float)
                            if 'TARGET AF' not in df_hist.columns:
                                results.append({'Komponen': col, 'mi': np.nan, 'Intercept': np.nan, 'R2': np.nan})
                                continue
                            X = sm.add_constant(df_hist['TARGET AF'].astype(float))
                            model = sm.OLS(y, X).fit()
                            results.append({'Komponen': col,'mi': model.params['TARGET AF'] if 'TARGET AF' in model.params.index else np.nan,'Intercept': model.params['const'] if 'const' in model.params.index else np.nan,'R2': model.rsquared})
                        except Exception:
                            results.append({'Komponen': col, 'mi': np.nan, 'Intercept': np.nan, 'R2': np.nan})

                    df_mi = pd.DataFrame(results)
                    df_sum = df_sum.merge(df_mi, on='Komponen', how='left')

                    TargetNPBT_2025 = df_sum['TargetOld'].sum()
                    Tax = TargetNPBT_2025 * 0.22 * (-1)
                    TargetNPATOld = TargetNPBT_2025 + Tax

                    df_out = compute_targets_2026(df_sum, TargetNPATOld=TargetNPATOld, TargetNPATNew=float(TargetNPATNew), k=None)
                    A_alloc_nasional = df_out[["Komponen","TargetNew"]].copy()
                    df_growth = df_out[["Komponen","TargetOld","TargetNew"]].copy()
                    df_growth["Growth_%"] = (df_growth["TargetNew"] - df_growth["TargetOld"]) / df_growth["TargetOld"] * 100


                    # ==========================================
                    # Correlation heatmap: specified columns vs 'TARGET MTD'
                    # Letakkan tepat setelah: st.dataframe(df_hist_filtered)
                    # ==========================================
                    import matplotlib.pyplot as plt

                    cols_corr = [
                        "Cons Fin","Titipan", "Pend. Adm","COF","Biaya Mktng","Biaya Mktng_2",
                        "Biaya Kary","Bbn Adm. Umum", "Pend. Denda", "Pend. PT&PP - Others", "Recovery Inc.",
                        "Pend. Lain-lain", "Pend. Asuransi Siaga", "WO", "WO Pelsus","Add. Prov WO",
                        "LOR", "Add. Prov LOR", "Penalty LOR","Biaya Denda+Rec+Lor"
                    ]

                    # pastikan df_hist_filtered sudah ada (hasil filter tahun yang dipilih)
                    # periksa kolom yang tersedia
                    available = [c for c in cols_corr if c in df_hist.columns]
                    missing = [c for c in cols_corr if c not in df_hist.columns]

                    if 'Target NPAT' not in df_hist.columns:
                        st.error(df_hist.columns)
                    else:
                        if len(available) == 0:
                            st.error("Tidak ada kolom komponen yang cocok ditemukan di data historis untuk dibuat korelasinya.")
                        else:
                            # Buat dataframe korelasi yang hanya berisi kolom yg tersedia + TARGET MTD
                            df_corr = df_hist[available + ['Target NPAT']].copy()

                            # Pastikan numeric (non-numeric => NaN), lalu drop rows/cols yang seluruhnya NaN
                            df_corr = df_corr.apply(pd.to_numeric, errors='coerce')
                            df_corr = df_corr.dropna(axis=0, how='all')   # drop baris semua NaN
                            df_corr = df_corr.dropna(axis=1, how='all')   # drop kolom semua NaN (jika ada)

                            # jika setelah pembersihan tidak ada data cukup
                            if df_corr.shape[0] < 2:
                                st.warning("Data historis terlalu sedikit setelah pembersihan (butuh >=2 baris) — heatmap tidak relevan.")
                            else:
                                corr_matrix = df_corr.corr()

                                # Plot heatmap (matplotlib)
                                fig, ax = plt.subplots(figsize=(10, max(6, len(corr_matrix)*0.35)))
                                cax = ax.imshow(corr_matrix, vmin=-1, vmax=1, aspect='auto', cmap='coolwarm')
                                ax.set_xticks(range(len(corr_matrix.columns)))
                                ax.set_yticks(range(len(corr_matrix.index)))
                                ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
                                ax.set_yticklabels(corr_matrix.index)

                                # annotate angka korelasi di tiap cell
                                for (i, j), val in np.ndenumerate(corr_matrix.values):
                                    ax.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=8, color='white' if abs(val)>0.5 else 'black')

                                fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
                                st.subheader("🔎 Correlation Heatmap (komponen vs TARGET NPAT)")
                                st.pyplot(fig)

                                # # tampilkan juga matriks korelasi sebagai tabel (opsional)
                                # st.subheader("Matriks Korelasi (angka)")
                                # st.dataframe(corr_matrix.style.format("{:.2f}"))

                                # beri info kolom yg hilang
                                if missing:
                                    st.warning(f"Beberapa kolom yang diminta tidak ditemukan di data historis: {missing}")


                    st.success("Perhitungan Nasional selesai.")
                    st.subheader("Alokasi Nasional")
                    st.dataframe(A_alloc_nasional)

                    # ------------------
                    # AREA ALLOCATION
                    # ------------------
                    try:
                        df_hist_filtered = df_hist2[df_hist2["Tahun"].isin([2024])].copy() if 'Tahun' in df_hist2.columns else df_hist2.copy()
                        if 'BULAN' in df_hist_filtered.columns:
                            df_hist_filtered = df_hist_filtered[df_hist_filtered["BULAN"] != 202507]
                        if 'ID CAB' in df_hist_filtered.columns:
                            df_hist_filtered = df_hist_filtered[~df_hist_filtered["ID CAB"].isin([98000, 99900])]
                        df_hist_filtered.columns = df_hist_filtered.columns.str.strip()

                        selected_cols = ["KEY","BULAN","Tahun","ID CAB","Cabang","Area","NPAT MTD","TARGET MTD","NSA","TARGET AF"]
                        selected_cols = [c for c in selected_cols if c in df_hist_filtered.columns]
                        df_hist_sel = df_hist_filtered[selected_cols].copy()
                        cols_to_str = [c for c in ["KEY","Tahun","ID CAB","BULAN"] if c in df_hist_sel.columns]
                        df_hist_sel[cols_to_str] = df_hist_sel[cols_to_str].astype(str)

                        for col in ["NPAT MTD","TARGET MTD","NSA","TARGET AF"]:
                            if col in df_hist_sel.columns:
                                df_hist_sel[col] = pd.to_numeric(df_hist_sel[col], errors='coerce')

                        numeric_cols = df_hist_sel.select_dtypes(include='number').columns.tolist()
                        df_pivotArea = df_hist_sel.groupby("Area", as_index=False)[numeric_cols].sum()
                        if 'NPAT MTD' in df_pivotArea.columns and 'TARGET MTD' in df_pivotArea.columns:
                            df_pivotArea['AchNPAT(%)'] = ((df_pivotArea['NPAT MTD'] - df_pivotArea['TARGET MTD']) / df_pivotArea['TARGET MTD']) * 100

                        df_histTarget = df_histTarget[pd.to_numeric(df_histTarget['Periode'], errors='coerce').notna()] if 'Periode' in df_histTarget.columns else df_histTarget.copy()
                        cols_to_str2 = [c for c in ['Periode','Branch'] if c in df_histTarget.columns]
                        df_histTarget[cols_to_str2] = df_histTarget[cols_to_str2].astype(str)
                        numeric_cols2 = df_histTarget.select_dtypes(include='number').columns.tolist()
                        if 'WILAYAH' in df_histTarget.columns:
                            df_pivotHistTargetArea = df_histTarget.groupby('WILAYAH', as_index=False)[numeric_cols2].sum()
                        elif 'Area' in df_histTarget.columns:
                            df_pivotHistTargetArea = df_histTarget.groupby('Area', as_index=False)[numeric_cols2].sum()
                        else:
                            df_pivotHistTargetArea = pd.DataFrame()

                        drop_cols = [c for c in ['Gross Profit','Opr Profit','NPBT','TAX','TARGET NPAT 2025','Pro NSA 2025'] if c in df_pivotHistTargetArea.columns]
                        if drop_cols:
                            df_pivotHistTargetArea = df_pivotHistTargetArea.drop(columns=drop_cols)
                        df_pivotHistTargetArea.columns = df_pivotHistTargetArea.columns.str.strip()

                        if not df_pivotHistTargetArea.empty:
                            metric_cols = [c for c in df_histTarget.columns if c not in ["Periode","REGION","WILAYAH","Cabang","Branch"]]
                            for c in metric_cols:
                                df_histTarget[c] = pd.to_numeric(df_histTarget[c], errors='coerce')
                            df_pivotHistTargetCabang = pd.pivot_table(df_histTarget, index=["WILAYAH", "Cabang"], values=metric_cols, aggfunc="sum", fill_value=0).reset_index()
                            df_pivotHistTargetCabang = df_pivotHistTargetCabang.rename(columns={"WILAYAH": "Area"})
                            drop_cols2 = [c for c in ['Gross Profit','Opr Profit','NPBT','TAX','TARGET NPAT 2025','Pro NSA 2025'] if c in df_pivotHistTargetCabang.columns]
                            if drop_cols2:
                                df_pivotHistTargetCabang = df_pivotHistTargetCabang.drop(columns=drop_cols2)
                            df_pivotHistTargetCabang.columns = df_pivotHistTargetCabang.columns.str.strip()

                        # call allocator (area level)
                        if df_pivotHistTargetArea.empty:
                            st.warning("Tidak cukup data untuk membuat pivot area historis. Alokasi area dilewatkan.")
                        else:
                            A_alloc_area, df_growth_area, diagnostics_area = allocate_npat(df_growth, df_pivotArea, df_pivotHistTargetArea)

                            # build A0 baseline
                            common_areas = A_alloc_area.index.tolist()
                            components = df_growth['Komponen'].tolist()
                            A0 = pd.DataFrame(index=common_areas, columns=components, dtype=float)
                            for c in components:
                                if c in df_pivotHistTargetArea.columns:
                                    A0[c] = df_pivotHistTargetArea.set_index('WILAYAH' if 'WILAYAH' in df_pivotHistTargetArea.columns else 'Area').reindex(common_areas)[c].astype(float).values
                                else:
                                    A0[c] = 0.0

                            df_delta = A_alloc_area - A0
                            df_pct_alloc = None
                            try:
                                df_pct_alloc = pd.DataFrame()
                                def compute_allocation_percentage(A_alloc, A0, df_growth):
                                    components = df_growth['Komponen'].tolist()
                                    df_pct = pd.DataFrame(index=A_alloc.index, columns=components, dtype=float)
                                    for c in components:
                                        if c not in A_alloc.columns:
                                            continue
                                        row = df_growth.loc[df_growth['Komponen'] == c]
                                        if row.empty:
                                            continue
                                        delta_total = float(row['TargetNew'].values[0]) - float(row['TargetOld'].values[0])
                                        if abs(delta_total) < EPS:
                                            df_pct[c] = 0.0
                                            continue
                                        delta_area = A_alloc[c].astype(float) - A0[c].astype(float)
                                        df_pct[c] = delta_area / delta_total * 100.0
                                    df_pct.index.name = 'Area'
                                    return df_pct

                                df_pct_alloc = compute_allocation_percentage(A_alloc_area, A0, df_growth)
                            except Exception:
                                df_pct_alloc = None

                            st.subheader("Alokasi per Area")
                            st.dataframe(A_alloc_area)

                            if df_pct_alloc is not None:
                                st.subheader("Persentase Growth per Area (%)")
                                st.dataframe(df_growth_area)

                            # ------------------
                            # CABANG ALLOCATION
                            # ------------------
                            # prepare df_pivotCabang
                            try:
                                if 'Area' not in df_hist_sel.columns or 'Cabang' not in df_hist_sel.columns:
                                    # try to build df_pivotCabang from df_pivotHistTargetCabang if available
                                    if 'df_pivotHistTargetCabang' in locals():
                                        df_pivotCabang = df_pivotHistTargetCabang.rename(columns={'WILAYAH': 'Area'}) if 'df_pivotHistTargetCabang' in locals() else pd.DataFrame()
                                    else:
                                        df_pivotCabang = pd.DataFrame()
                                else:
                                    df_pivotCabang = pd.pivot_table(df_hist_sel, index=['Area','Cabang'], values=['NPAT MTD','TARGET MTD','NSA','TARGET AF'], aggfunc='sum', fill_value=0).reset_index()

                                if df_pivotCabang.empty:
                                    st.warning("Tidak cukup data untuk membuat pivot cabang. Alokasi cabang dilewatkan.")
                                else:
                                    A_alloc_cabang, df_growth_cabang, diagnostics_cab = allocate_area_to_cabang(A_alloc_area, df_pivotCabang, df_pivotHistTargetCabang if 'df_pivotHistTargetCabang' in locals() else None, df_growth_area)

                                    st.subheader("Alokasi per Cabang")
                                    st.dataframe(A_alloc_cabang)

                                    st.subheader("Growth per Cabang (%)")
                                    st.dataframe(df_growth_cabang)

                                    # aggregate back to area to validate
                                    A_alloc_area_from_cabang = A_alloc_cabang.groupby(level='Area').sum()
                                    A_alloc_area_from_cabang.index.name = 'Area'

                                    # st.subheader("(Check) Rekap Alokasi Area dari Hasil Cabang (sum)")
                                    # st.dataframe(A_alloc_area_from_cabang)

                                try: 
                                    use_cached_models = st.checkbox("Gunakan model yang tersimpan di sesi (jika ada) — hindari re-train", value=True)
                                    if any(v is None for v in [A_alloc_nasional, A_alloc_area, A_alloc_cabang]):
                                        st.error("A_alloc_nasional / A_alloc_area / A_alloc_cabang belum tersedia. Jalankan tahap sebelumnya dulu.")
                                    else:
                                        with st.spinner("Menyiapkan data contour dari df_full_hist ..."):
                                            try:
                                                # 1) buat df_BahanContour sesuai instruksi Anda
                                                df_BahanContour = df_full_hist[["Tahun","BULAN","Pendapatan PK Gross", "TOTAL", "TOTAL39"]].copy()

                                                df_BahanContour = df_BahanContour.dropna(
                                                    subset=["Pendapatan PK Gross", "TOTAL", "TOTAL39"]
                                                )

                                                df_BahanContour["BULAN"] = df_BahanContour["BULAN"].astype(int).astype(str)

                                                df_BahanContour = (
                                                    df_BahanContour
                                                    .groupby("BULAN", as_index=False)
                                                    .agg({
                                                        "Tahun": "first",
                                                        "Pendapatan PK Gross": "sum",
                                                        "TOTAL": "sum",
                                                        "TOTAL39": "sum",
                                                    })
                                                )

                                                last_month = df_BahanContour["BULAN"].max()
                                                df_BahanContour = df_BahanContour[df_BahanContour["BULAN"] != last_month]

                                                # jumlah hari
                                                df_BahanContour["Jumlah Hari"] = df_BahanContour.apply(
                                                    lambda row: days_in_month(row["Tahun"], int(str(row["BULAN"])[-2:])),
                                                    axis=1
                                                )

                                                # ekstrak bulan & flags
                                                df_BahanContour["Bulan_Num"] = df_BahanContour["BULAN"].astype(str).str[-2:].astype(int)
                                                df_BahanContour["IdulFitri"] = df_BahanContour.apply(
                                                    lambda row: is_idulfitri(row["Tahun"], row["Bulan_Num"]), axis=1
                                                )
                                                df_BahanContour["Natal"] = df_BahanContour["Bulan_Num"].apply(is_natal)
                                                df_BahanContour["TahunBaru"] = df_BahanContour["Bulan_Num"].apply(is_tahunbaru)
                                                df_BahanContour["BulanPendek"] = df_BahanContour["Bulan_Num"].apply(is_bulanpendek)

                                                st.success("df_BahanContour siap.")
                                            except Exception as e:
                                                st.error(f"Error menyiapkan df_BahanContour: {e}")
                                                st.stop()

                                        # ----- Model 1 (prop NPAT) -----
                                        with st.spinner("(Model1) Melatih/menjalankan train_xgb_prop ..."):
                                            try:
                                                # gunakan session_state untuk menyimpan model agar tidak retrain tiap interaksi
                                                cached1 = st.session_state.get("contour_model_1", None)
                                                cached1_meta = st.session_state.get("contour_model_1_meta", None)

                                                # buat fingerprint sederhana untuk memastikan kecocokan data (shape + sum)
                                                fp1 = (df_BahanContour.shape, float(df_BahanContour["Pendapatan PK Gross"].sum()))

                                                if use_cached_models and cached1 is not None and cached1_meta == fp1:
                                                    model, feats, train_res, valid_res = cached1
                                                    # st.info("Model1: menggunakan model tersimpan di session_state.")
                                                else:
                                                    dfp = prepare_df(df_BahanContour)
                                                    model, feats, train_res, valid_res = train_xgb_prop(dfp, train_years=[2017,2018,2019,2021,2022], valid_years=[2023])
                                                    st.session_state["contour_model_1"] = (model, feats, train_res, valid_res)
                                                    st.session_state["contour_model_1_meta"] = fp1
                                                    # st.success("Model1: selesai dilatih dan disimpan pada session_state.")
                                            except Exception as e:
                                                st.error(f"Error di train_xgb_prop: {e}")
                                                st.stop()

                                        # ----- Model 2 (coll ratio) -----
                                        with st.spinner("(Model2) Melatih/menjalankan train_xgb_prop2 ..."):
                                            try:
                                                df_BahanContour2 = df_BahanContour.copy()
                                                df_BahanContour2["coll"] = df_BahanContour2["TOTAL39"] / df_BahanContour2["TOTAL"]

                                                cached2 = st.session_state.get("contour_model_2", None)
                                                cached2_meta = st.session_state.get("contour_model_2_meta", None)
                                                fp2 = (df_BahanContour2.shape, float(df_BahanContour2["coll"].sum()))

                                                if use_cached_models and cached2 is not None and cached2_meta == fp2:
                                                    model_2, feats_2, train_res_2, valid_res_2 = cached2
                                                    # st.info("Model2: menggunakan model tersimpan di session_state.")
                                                else:
                                                    dfp2 = prepare_df2(df_BahanContour2)
                                                    model_2, feats_2, train_res_2, valid_res_2 = train_xgb_prop2(dfp2, train_years=[2017,2018,2019,2021,2022], valid_years=[2023])
                                                    st.session_state["contour_model_2"] = (model_2, feats_2, train_res_2, valid_res_2)
                                                    st.session_state["contour_model_2_meta"] = fp2
                                                    # st.success("Model2: selesai dilatih dan disimpan pada session_state.")
                                            except Exception as e:
                                                st.error(f"Error di train_xgb_prop2: {e}")
                                                st.stop()

                                        # ----- Predict & buat contour -----
                                        with st.spinner("Menghasilkan contour untuk 2026 dan midpoint ..."):
                                            try:
                                                pred2026 = predict_prop(2026, model, feats, idul_month=3)
                                                contour2026 = get_contour(2026, model, feats, idul_month=3)
                                                contour2026 = pd.DataFrame(contour2026)

                                                pred2026_2 = predict_prop2(2026, model_2, feats_2, idul_month=3)
                                                contour2026_2 = get_contour2(2026, model_2, feats_2, idul_month=3)
                                                contour2026_2 = pd.DataFrame(contour2026_2)

                                                contour_new = midpoint_contour(contour2026, contour2026_2)  # kolom month & contour_mid

                                                # st.success("Contour (midpoint) siap.")
                                            except Exception as e:
                                                st.error(f"Error membuat contour/prediksi: {e}")
                                                st.stop()

                                        # ----- Alokasikan dengan contour -----
                                        with st.spinner("Melakukan allocate_with_contour (menghasilkan bulanan)..."):
                                            try:
                                                # allocate_with_contour Anda mengembalikan dict dan menulis file (opsional)
                                                # Kita akan ambil hasilnya sebagai DataFrames
                                                results = allocate_with_contour(
                                                    A_alloc_nasional,
                                                    A_alloc_area,
                                                    A_alloc_cabang,
                                                    contour_new,
                                                    output_file=None  # jika fungsi Anda memerlukan nama file, Anda bisa mengirimkan path; 
                                                                    # jika tidak, kita tetap pakai hasil yang dikembalikan
                                                )
                                                # Jika allocate_with_contour milik Anda menulis ke disk dan tidak mengembalikan dict,
                                                # Anda bisa memanggilnya lalu membangun Excel sendiri. Asumsi di sini: mengembalikan dict.
                                            except Exception as e:
                                                # fallback: jika fungsi Anda mengharapkan output_file dan menulis file,
                                                # panggil dengan nama file lalu load file itu. Namun preference: fungsi mengembalikan dict.
                                                try:
                                                    results = allocate_with_contour(
                                                        A_alloc_nasional,
                                                        A_alloc_area,
                                                        A_alloc_cabang,
                                                        contour_new,
                                                        output_file="alokasi_output_contour.xlsx"
                                                    )
                                                    # st.info("allocate_with_contour menulis file alokasi_output_contour.xlsx")
                                                except Exception as e2:
                                                    st.error(f"Gagal menjalankan allocate_with_contour: {e} | fallback error: {e2}")
                                                    st.stop()

                                        # ----- Tampilkan hasil & download ----

                                        # ==========================================
                                        # Line Chart Perbandingan Contour
                                        # ==========================================
                                        import matplotlib.pyplot as plt

                                        fig, ax = plt.subplots(figsize=(16,5))

                                        ax.plot(contour2026["month"], contour2026["contour"], marker='o', label="Contour Cons Fin")
                                        ax.plot(contour2026_2["month"], contour2026_2["contour"], marker='s', label="Contour Coll")
                                        ax.plot(contour_new["month"], contour_new["contour_mid"], marker='^', label="Contour Mid")

                                        ax.set_xlabel("Month")
                                        ax.set_ylabel("Contour Value")
                                        ax.set_title("Perbandingan Contour")
                                        ax.legend()
                                        ax.grid(True)

                                        st.subheader("📈 Perbandingan Contour (Line Chart)")
                                        st.pyplot(fig)


                                        st.subheader("Hasil Contour (ConsFin)")
                                        st.dataframe(contour2026)

                                        st.subheader("Hasil Contour (Coll)")
                                        st.dataframe(contour2026_2)

                                        st.subheader("Hasil Contour (Mid)")
                                        st.dataframe(contour_new)

                                        st.subheader("Hasil Alokasi Bulanan")
                                        # tampilkan ringkasan (jumlah per sheet)
                                        try:
                                            nas_fy = results["nasional_fy"]
                                            nas_bulan = results["nasional_bulanan"]
                                            area_fy = results["area_fy"]
                                            area_bulan = results["area_bulanan"]
                                            cab_fy = results["cabang_fy"]
                                            cab_bulan = results["cabang_bulanan"]

                                            # st.write("Nasional (Full Year) — sample:")
                                            # st.dataframe(nas_fy.head())

                                            st.write("Nasional Bulanan")
                                            st.dataframe(nas_bulan)

                                            # st.write("Area (Full Year) — sample:")
                                            # st.dataframe(area_fy)

                                            st.write("Area Bulanan")
                                            st.dataframe(area_bulan.reset_index())

                                            st.write("Cabang Bulanan")
                                            st.dataframe(cab_bulan.reset_index())
                                        except Exception:
                                            # jika struktur berbeda, dump keys
                                            st.write("Results keys:", results.keys())
                                            for k,v in results.items():
                                                st.write(k)
                                                try:
                                                    st.dataframe(v.head())
                                                except Exception:
                                                    st.write(type(v))

                                        # Buat Excel in-memory dan sediakan download button
                                        with st.spinner("Menyiapkan file Excel untuk diunduh ..."):
                                            try:
                                                towrite = io.BytesIO()
                                                with pd.ExcelWriter(towrite, engine="xlsxwriter") as writer:
                                                    # sheet names singkat & rapi
                                                    nas_fy.to_excel(writer, sheet_name="Nasional_FY", index=False)
                                                    nas_bulan.to_excel(writer, sheet_name="Nasional_Bulanan", index=False)

                                                    # area_fy mungkin index Area
                                                    area_fy.to_excel(writer, sheet_name="Area_FY")
                                                    # area_bulanan bisa MultiIndex; reset index agar tersimpan
                                                    area_bulan.reset_index().to_excel(writer, sheet_name="Area_Bulanan", index=False)

                                                    cab_fy.reset_index().to_excel(writer, sheet_name="Cabang_FY", index=False)
                                                    cab_bulan.reset_index().to_excel(writer, sheet_name="Cabang_Bulanan", index=False)

                                                    # juga simpan contour & model diagnostics simple
                                                    contour_new.to_excel(writer, sheet_name="Contour_Midpoint", index=False)

                                                towrite.seek(0)
                                                st.download_button("Download semua hasil (Excel)", data=towrite, file_name="alokasi_bulanan_all.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                                                st.success("File Excel siap diunduh.")
                                            except Exception as e:
                                                st.error(f"Gagal membuat file Excel: {e}")

                                        st.success("Tahap Contour & Alokasi Bulanan selesai.")
                                except Exception as e:
                                    st.warning(f"Gagal melakukan alokasi cabang: {e}")

                                    # downloads
                                    towrite = io.BytesIO()
                                    with pd.ExcelWriter(towrite, engine='xlsxwriter') as writer:
                                        df_out.to_excel(writer, sheet_name='df_out', index=False)
                                        A_alloc_nasional.to_excel(writer, sheet_name='A_alloc_nasional', index=False)
                                        A_alloc_area.to_excel(writer, sheet_name='A_alloc_area')
                                        if 'df_pct_alloc' in locals() and df_pct_alloc is not None:
                                            df_pct_alloc.to_excel(writer, sheet_name='pct_alloc')
                                        A_alloc_cabang.to_excel(writer, sheet_name='A_alloc_cabang')
                                        df_growth_cabang.to_excel(writer, sheet_name='df_growth_cabang')
                                    towrite.seek(0)
                                    st.download_button("Download Semua Hasil (Excel)", data=towrite, file_name='hasil_nasional_area_cabang.xlsx')

                            except Exception as e:
                                st.warning(f"Gagal melakukan alokasi cabang: {e}")

                    except Exception as e:
                        st.error(f"Gagal melakukan alokasi area/cabang: {e}")

                except Exception as e:
                    st.error(f"Error saat perhitungan nasional: {e}")

            else:
                st.info("Siap: upload file lalu klik tombol untuk menjalankan perhitungan penuh.")

            # seluruh pipeline perhitungan target
        except Exception as e:
            st.error(f"Gagal membaca file atau sheet: {e}")
    



# ============================
# TAB 2: MULAI DARI TARGET AREA FY
# ============================
with tab2:
    st.markdown("### 🚀 Jalankan hanya dari *Target Area Full Year* (skip tahap Nasional)")

    st.info("""
    Unggah **satu file Excel** yang berisi tiga sheet:
    1️⃣ Komponen Profit Y-1  
    2️⃣ Data Historis  
    3️⃣ Target Area FY per komponen  
    """)

    file_excel = st.file_uploader("📘 Unggah file Excel (semua sheet di dalam satu file)", type=["xls", "xlsx"], key="excel_area_only")

    if file_excel:
        try:
            # Baca daftar sheet
            xls = pd.ExcelFile(file_excel)
            sheets = xls.sheet_names

            # Pilih 3 sheet dari file yang sama
            col1, col2, col3 = st.columns(3)
            with col1:
                sheet_y1 = st.selectbox("Sheet Komponen Profit Y-1", sheets, index=0)
            with col2:
                sheet_hist = st.selectbox("Sheet Data Historis", sheets, index=min(1, len(sheets)-1))
            with col3:
                sheet_area = st.selectbox("Sheet Target Area FY", sheets, index=min(2, len(sheets)-1))

            # Baca ketiga sheet yang dipilih
            df_y1 = pd.read_excel(file_excel, sheet_name=sheet_y1)
            df_full_hist = pd.read_excel(file_excel, sheet_name=sheet_hist, header=3)
            area_fy = pd.read_excel(file_excel, sheet_name=sheet_area, index_col=0)

            st.success("✅ Semua sheet berhasil dibaca dari file Excel.")
            st.write("**Preview: Target Area FY**")
            st.dataframe(area_fy.head())

            try:
                df_histTarget = df_y1.copy()
                df_hist2 = df_full_hist.copy()
                A_alloc_area = area_fy.copy()

                df_hist_filtered = df_hist2[df_hist2["Tahun"].isin([2024])].copy() if 'Tahun' in df_hist2.columns else df_hist2.copy()
                if 'BULAN' in df_hist_filtered.columns:
                    df_hist_filtered = df_hist_filtered[df_hist_filtered["BULAN"] != 202507]
                if 'ID CAB' in df_hist_filtered.columns:
                    df_hist_filtered = df_hist_filtered[~df_hist_filtered["ID CAB"].isin([98000, 99900])]
                df_hist_filtered.columns = df_hist_filtered.columns.str.strip()

                selected_cols = ["KEY","BULAN","Tahun","ID CAB","Cabang","Area","NPAT MTD","TARGET MTD","NSA","TARGET AF"]
                selected_cols = [c for c in selected_cols if c in df_hist_filtered.columns]
                df_hist_sel = df_hist_filtered[selected_cols].copy()
                cols_to_str = [c for c in ["KEY","Tahun","ID CAB","BULAN"] if c in df_hist_sel.columns]
                df_hist_sel[cols_to_str] = df_hist_sel[cols_to_str].astype(str)

                for col in ["NPAT MTD","TARGET MTD","NSA","TARGET AF"]:
                    if col in df_hist_sel.columns:
                        df_hist_sel[col] = pd.to_numeric(df_hist_sel[col], errors='coerce')

                numeric_cols = df_hist_sel.select_dtypes(include='number').columns.tolist()
                df_pivotArea = df_hist_sel.groupby("Area", as_index=False)[numeric_cols].sum()
                if 'NPAT MTD' in df_pivotArea.columns and 'TARGET MTD' in df_pivotArea.columns:
                    df_pivotArea['AchNPAT(%)'] = ((df_pivotArea['NPAT MTD'] - df_pivotArea['TARGET MTD']) / df_pivotArea['TARGET MTD']) * 100

                df_histTarget = df_histTarget[pd.to_numeric(df_histTarget['Periode'], errors='coerce').notna()] if 'Periode' in df_histTarget.columns else df_histTarget.copy()
                cols_to_str2 = [c for c in ['Periode','Branch'] if c in df_histTarget.columns]
                df_histTarget[cols_to_str2] = df_histTarget[cols_to_str2].astype(str)
                numeric_cols2 = df_histTarget.select_dtypes(include='number').columns.tolist()
                if 'WILAYAH' in df_histTarget.columns:
                    df_pivotHistTargetArea = df_histTarget.groupby('WILAYAH', as_index=False)[numeric_cols2].sum()
                elif 'Area' in df_histTarget.columns:
                    df_pivotHistTargetArea = df_histTarget.groupby('Area', as_index=False)[numeric_cols2].sum()
                else:
                    df_pivotHistTargetArea = pd.DataFrame()

                drop_cols = [c for c in ['Gross Profit','Opr Profit','NPBT','TAX','TARGET NPAT 2025','Pro NSA 2025'] if c in df_pivotHistTargetArea.columns]
                if drop_cols:
                    df_pivotHistTargetArea = df_pivotHistTargetArea.drop(columns=drop_cols)
                df_pivotHistTargetArea.columns = df_pivotHistTargetArea.columns.str.strip()

                if not df_pivotHistTargetArea.empty:
                    metric_cols = [c for c in df_histTarget.columns if c not in ["Periode","REGION","WILAYAH","Cabang","Branch"]]
                    for c in metric_cols:
                        df_histTarget[c] = pd.to_numeric(df_histTarget[c], errors='coerce')
                    df_pivotHistTargetCabang = pd.pivot_table(df_histTarget, index=["WILAYAH", "Cabang"], values=metric_cols, aggfunc="sum", fill_value=0).reset_index()
                    df_pivotHistTargetCabang = df_pivotHistTargetCabang.rename(columns={"WILAYAH": "Area"})
                    drop_cols2 = [c for c in ['Gross Profit','Opr Profit','NPBT','TAX','TARGET NPAT 2025','Pro NSA 2025'] if c in df_pivotHistTargetCabang.columns]
                    if drop_cols2:
                        df_pivotHistTargetCabang = df_pivotHistTargetCabang.drop(columns=drop_cols2)
                    df_pivotHistTargetCabang.columns = df_pivotHistTargetCabang.columns.str.strip()

                if 'Area' not in df_hist_sel.columns or 'Cabang' not in df_hist_sel.columns:
                    # try to build df_pivotCabang from df_pivotHistTargetCabang if available
                    if 'df_pivotHistTargetCabang' in locals():
                        df_pivotCabang = df_pivotHistTargetCabang.rename(columns={'WILAYAH': 'Area'}) if 'df_pivotHistTargetCabang' in locals() else pd.DataFrame()
                    else:
                        df_pivotCabang = pd.DataFrame()
                else:
                    df_pivotCabang = pd.pivot_table(df_hist_sel, index=['Area','Cabang'], values=['NPAT MTD','TARGET MTD','NSA','TARGET AF'], aggfunc='sum', fill_value=0).reset_index()

                if df_pivotCabang.empty:
                    st.warning("Tidak cukup data untuk membuat pivot cabang. Alokasi cabang dilewatkan.")
                else:
                    A_alloc_cabang, df_growth_cabang, diagnostics_cab = allocate_area_to_cabang(A_alloc_area, df_pivotCabang, df_pivotHistTargetCabang if 'df_pivotHistTargetCabang' in locals() else None, A_alloc_area)

                    st.subheader("Alokasi per Cabang")
                    st.dataframe(A_alloc_cabang)

                    st.subheader("Growth per Cabang (%)")
                    st.dataframe(df_growth_cabang)

                    # aggregate back to area to validate
                    A_alloc_area_from_cabang = A_alloc_cabang.groupby(level='Area').sum()
                    A_alloc_area_from_cabang.index.name = 'Area'

                    # st.subheader("(Check) Rekap Alokasi Area dari Hasil Cabang (sum)")
                    # st.dataframe(A_alloc_area_from_cabang)

                try: 
                    use_cached_models = st.checkbox("Gunakan model yang tersimpan di sesi (jika ada) — hindari re-train", value=True)
                    if any(v is None for v in [A_alloc_area, A_alloc_cabang]):
                        st.error("A_alloc_area / A_alloc_cabang belum tersedia. Jalankan tahap sebelumnya dulu.")
                    else:
                        with st.spinner("Menyiapkan data contour dari df_full_hist ..."):
                            try:
                                # 1) buat df_BahanContour sesuai instruksi Anda
                                df_BahanContour = df_full_hist[["Tahun","BULAN","Pendapatan PK Gross", "TOTAL", "TOTAL39"]].copy()

                                df_BahanContour = df_BahanContour.dropna(
                                    subset=["Pendapatan PK Gross", "TOTAL", "TOTAL39"]
                                )

                                df_BahanContour["BULAN"] = df_BahanContour["BULAN"].astype(int).astype(str)

                                df_BahanContour = (
                                    df_BahanContour
                                    .groupby("BULAN", as_index=False)
                                    .agg({
                                        "Tahun": "first",
                                        "Pendapatan PK Gross": "sum",
                                        "TOTAL": "sum",
                                        "TOTAL39": "sum",
                                    })
                                )

                                last_month = df_BahanContour["BULAN"].max()
                                df_BahanContour = df_BahanContour[df_BahanContour["BULAN"] != last_month]

                                # jumlah hari
                                df_BahanContour["Jumlah Hari"] = df_BahanContour.apply(
                                    lambda row: days_in_month(row["Tahun"], int(str(row["BULAN"])[-2:])),
                                    axis=1
                                )

                                # ekstrak bulan & flags
                                df_BahanContour["Bulan_Num"] = df_BahanContour["BULAN"].astype(str).str[-2:].astype(int)
                                df_BahanContour["IdulFitri"] = df_BahanContour.apply(
                                    lambda row: is_idulfitri(row["Tahun"], row["Bulan_Num"]), axis=1
                                )
                                df_BahanContour["Natal"] = df_BahanContour["Bulan_Num"].apply(is_natal)
                                df_BahanContour["TahunBaru"] = df_BahanContour["Bulan_Num"].apply(is_tahunbaru)
                                df_BahanContour["BulanPendek"] = df_BahanContour["Bulan_Num"].apply(is_bulanpendek)

                                st.success("df_BahanContour siap.")
                            except Exception as e:
                                st.error(f"Error menyiapkan df_BahanContour: {e}")
                                st.stop()

                        # ----- Model 1 (prop NPAT) -----
                        with st.spinner("(Model1) Melatih/menjalankan train_xgb_prop ..."):
                            try:
                                # gunakan session_state untuk menyimpan model agar tidak retrain tiap interaksi
                                cached1 = st.session_state.get("contour_model_1", None)
                                cached1_meta = st.session_state.get("contour_model_1_meta", None)

                                # buat fingerprint sederhana untuk memastikan kecocokan data (shape + sum)
                                fp1 = (df_BahanContour.shape, float(df_BahanContour["Pendapatan PK Gross"].sum()))

                                if use_cached_models and cached1 is not None and cached1_meta == fp1:
                                    model, feats, train_res, valid_res = cached1
                                    # st.info("Model1: menggunakan model tersimpan di session_state.")
                                else:
                                    dfp = prepare_df(df_BahanContour)
                                    model, feats, train_res, valid_res = train_xgb_prop(dfp, train_years=[2017,2018,2019,2021,2022], valid_years=[2023])
                                    st.session_state["contour_model_1"] = (model, feats, train_res, valid_res)
                                    st.session_state["contour_model_1_meta"] = fp1
                                    # st.success("Model1: selesai dilatih dan disimpan pada session_state.")
                            except Exception as e:
                                st.error(f"Error di train_xgb_prop: {e}")
                                st.stop()

                        # ----- Model 2 (coll ratio) -----
                        with st.spinner("(Model2) Melatih/menjalankan train_xgb_prop2 ..."):
                            try:
                                df_BahanContour2 = df_BahanContour.copy()
                                df_BahanContour2["coll"] = df_BahanContour2["TOTAL39"] / df_BahanContour2["TOTAL"]

                                cached2 = st.session_state.get("contour_model_2", None)
                                cached2_meta = st.session_state.get("contour_model_2_meta", None)
                                fp2 = (df_BahanContour2.shape, float(df_BahanContour2["coll"].sum()))

                                if use_cached_models and cached2 is not None and cached2_meta == fp2:
                                    model_2, feats_2, train_res_2, valid_res_2 = cached2
                                    # st.info("Model2: menggunakan model tersimpan di session_state.")
                                else:
                                    dfp2 = prepare_df2(df_BahanContour2)
                                    model_2, feats_2, train_res_2, valid_res_2 = train_xgb_prop2(dfp2, train_years=[2017,2018,2019,2021,2022], valid_years=[2023])
                                    st.session_state["contour_model_2"] = (model_2, feats_2, train_res_2, valid_res_2)
                                    st.session_state["contour_model_2_meta"] = fp2
                                    # st.success("Model2: selesai dilatih dan disimpan pada session_state.")
                            except Exception as e:
                                st.error(f"Error di train_xgb_prop2: {e}")
                                st.stop()

                        # ----- Predict & buat contour -----
                        with st.spinner("Menghasilkan contour untuk 2026 dan midpoint ..."):
                            try:
                                pred2026 = predict_prop(2026, model, feats, idul_month=3)
                                contour2026 = get_contour(2026, model, feats, idul_month=3)
                                contour2026 = pd.DataFrame(contour2026)

                                pred2026_2 = predict_prop2(2026, model_2, feats_2, idul_month=3)
                                contour2026_2 = get_contour2(2026, model_2, feats_2, idul_month=3)
                                contour2026_2 = pd.DataFrame(contour2026_2)

                                contour_new = midpoint_contour(contour2026, contour2026_2)  # kolom month & contour_mid

                                # st.success("Contour (midpoint) siap.")
                            except Exception as e:
                                st.error(f"Error membuat contour/prediksi: {e}")
                                st.stop()

                        # ----- Alokasikan dengan contour -----
                        with st.spinner("Melakukan allocate_with_contour (menghasilkan bulanan)..."):
                            try:
                                # allocate_with_contour Anda mengembalikan dict dan menulis file (opsional)
                                # Kita akan ambil hasilnya sebagai DataFrames
                                area_bulan = hitung_area_bulanan(A_alloc_area, contour_new)
                                cab_bulan = hitung_cabang_bulanan(A_alloc_cabang, contour_new)
                                # Jika allocate_with_contour milik Anda menulis ke disk dan tidak mengembalikan dict,
                                # Anda bisa memanggilnya lalu membangun Excel sendiri. Asumsi di sini: mengembalikan dict.
                            except Exception as e:
                                # fallback: jika fungsi Anda mengharapkan output_file dan menulis file,
                                # panggil dengan nama file lalu load file itu. Namun preference: fungsi mengembalikan dict.
                                try:
                                    area_bulan = hitung_area_bulanan(A_alloc_area, contour_new)
                                    cab_bulan = hitung_cabang_bulanan(A_alloc_cabang, contour_new)
                                    # st.info("allocate_with_contour menulis file alokasi_output_contour.xlsx")
                                except Exception as e2:
                                    st.error(f"Gagal menjalankan allocate_with_contour: {e} | fallback error: {e2}")
                                    st.stop()

                        # ----- Tampilkan hasil & download ----

                        # ==========================================
                        # Line Chart Perbandingan Contour
                        # ==========================================
                        import matplotlib.pyplot as plt

                        fig, ax = plt.subplots(figsize=(16,5))

                        ax.plot(contour2026["month"], contour2026["contour"], marker='o', label="Contour Cons Fin")
                        ax.plot(contour2026_2["month"], contour2026_2["contour"], marker='s', label="Contour Coll")
                        ax.plot(contour_new["month"], contour_new["contour_mid"], marker='^', label="Contour Mid")

                        ax.set_xlabel("Month")
                        ax.set_ylabel("Contour Value")
                        ax.set_title("Perbandingan Contour")
                        ax.legend()
                        ax.grid(True)

                        st.subheader("📈 Perbandingan Contour (Line Chart)")
                        st.pyplot(fig)


                        st.subheader("Hasil Contour (ConsFin)")
                        st.dataframe(contour2026)

                        st.subheader("Hasil Contour (Coll)")
                        st.dataframe(contour2026_2)

                        st.subheader("Hasil Contour (Mid)")
                        st.dataframe(contour_new)

                        st.subheader("Hasil Alokasi Bulanan")
                        # tampilkan ringkasan (jumlah per sheet)
                        # try:
                        #     area_fy = area_fy
                        #     area_bulan = area_bulan
                        #     cab_fy = cab_fy
                        #     cab_bulan = cab_bulan

                        #     # st.write("Nasional (Full Year) — sample:")
                        #     # st.dataframe(nas_fy.head())

                        #     st.write("Nasional Bulanan")
                        #     st.dataframe(nas_bulan)

                        #     # st.write("Area (Full Year) — sample:")
                        #     # st.dataframe(area_fy)

                        #     st.write("Area Bulanan")
                        #     st.dataframe(area_bulan.reset_index())

                        #     st.write("Cabang Bulanan")
                        #     st.dataframe(cab_bulan.reset_index())
                        # except Exception:
                        #     # jika struktur berbeda, dump keys
                        #     st.write("Results keys:", results.keys())
                        #     for k,v in results.items():
                        #         st.write(k)
                        #         try:
                        #             st.dataframe(v.head())
                        #         except Exception:
                        #             st.write(type(v))

                        # Buat Excel in-memory dan sediakan download button
                        # with st.spinner("Menyiapkan file Excel untuk diunduh ..."):
                        #     try:
                        #         towrite = io.BytesIO()
                        #         with pd.ExcelWriter(towrite, engine="xlsxwriter") as writer:
                        #             # sheet names singkat & rapi

                        #             # area_fy mungkin index Area
                        #             area_fy.to_excel(writer, sheet_name="Area_FY")
                        #             # area_bulanan bisa MultiIndex; reset index agar tersimpan
                        #             area_bulan.to_excel(writer, sheet_name="Area_Bulanan", index=False)

                        #             cab_fy.to_excel(writer, sheet_name="Cabang_FY", index=False)
                        #             cab_bulan.to_excel(writer, sheet_name="Cabang_Bulanan", index=False)

                        #             # juga simpan contour & model diagnostics simple
                        #             contour_new.to_excel(writer, sheet_name="Contour_Midpoint", index=False)

                        #         towrite.seek(0)
                        #         st.download_button("Download semua hasil (Excel)", data=towrite, file_name="alokasi_bulanan_all.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                        #         st.success("File Excel siap diunduh.")
                        #     except Exception as e:
                        #         st.error(f"Gagal membuat file Excel: {e}")

                        st.dataframe(area_bulan)
                        st.dataframe(cab_bulan)

                        st.success("Tahap Contour & Alokasi Bulanan selesai.")
                except Exception as e:
                    st.warning(f"Gagal melakukan alokasi cabang: {e}")

                st.divider()
                st.subheader("🤖 Analisis & Rekomendasi AI (SumoPod)")

                if st.button("🔍 Analisis dengan AI (Area Level)"):
                    with st.spinner("Mengirim data ke AI dan melakukan analisis..."):
                        payload_json = df_to_json_payload(A_alloc_area)
                        ai_result = call_sumopod_analysis(payload_json)

                    st.success("Analisis selesai")

                    st.markdown("### 🧠 Insight & Rekomendasi Strategis")
                    st.markdown(ai_result)
                    
                # downloads
                towrite = io.BytesIO()
                with pd.ExcelWriter(towrite, engine='xlsxwriter') as writer:
                    A_alloc_area.to_excel(writer, sheet_name='A_alloc_area')
                    if 'df_pct_alloc' in locals() and df_pct_alloc is not None:
                        df_pct_alloc.to_excel(writer, sheet_name='pct_alloc')
                    A_alloc_cabang.to_excel(writer, sheet_name='A_alloc_cabang')
                    df_growth_cabang.to_excel(writer, sheet_name='df_growth_cabang')
                    area_bulan.to_excel(writer, sheet_name='area_bulan')
                    cab_bulan.to_excel(writer, sheet_name='cab_bulan')
                towrite.seek(0)
                st.download_button("Download Semua Hasil (Excel)", data=towrite, file_name='hasil_nasional_area_cabang.xlsx')




            except Exception as e:
                st.warning(f"Gagal melakukan alokasi cabang: {e}")

        except Exception as e:
            st.error(f"Gagal menjalankan pipeline area-only: {e}")
    else:
        st.info("Unggah file Excel untuk memulai proses.")
