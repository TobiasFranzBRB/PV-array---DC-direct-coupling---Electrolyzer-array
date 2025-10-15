# app.py
# --------------------------------------------------------------------------------------
# Streamlit WebApp: PV Array ↔ Electrolyzer Coupling (PVGIS-driven, pvlib-based)
# --------------------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pvlib import iam
from pvlib.ivtools.sdm import fit_cec_sam
from pvlib import location, iotools, irradiance, temperature, pvsystem

# =========================
# Streamlit page config
# =========================
st.set_page_config(
    page_title="PV Array → direct DC coupling → Electrolyzer Array Coupling",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# Locations dictionary
# =========================
LOCATIONS = {
    "Seville_ES": {"lat": 37.3891, "lon": -5.9845, "alt": 7, "tz": "Europe/Madrid", "label": "Seville, Spain"},
    "Magdeburg_DE": {"lat": 52.1200, "lon": 11.6270, "alt": 55, "tz": "Europe/Berlin", "label": "Magdeburg, Germany"},
    "Lubbock_US": {"lat": 33.5779, "lon": -101.8552, "alt": 1000, "tz": "America/Chicago", "label": "Lubbock, Texas, USA"},
    "Luderitz_NA": {"lat": -26.6480, "lon": 15.1590, "alt": 30, "tz": "Africa/Windhoek", "label": "Lüderitz, Namibia"},
    "Riyadh_SA": {"lat": 24.7136, "lon": 46.6753, "alt": 612, "tz": "Asia/Riyadh", "label": "Riyadh, Saudi Arabia"},
    "Newman_AU": {"lat": -23.3560, "lon": 119.7350, "alt": 545, "tz": "Australia/Perth", "label": "Newman, Western Australia"},
    "Calama_CL": {"lat": -22.4540, "lon": -68.9290, "alt": 2260, "tz": "America/Santiago", "label": "Calama, Chile"},
}

# =========================
# Fixed/default parameters (NOT user inputs; set by code)
# =========================
CABLE_TEMP_C   = 45.0
ARR_CABLE_TEMP_C = 45.0
CROSS_SEC_MM2  = 4.0
MODULE_NAME = "Jinko_Solar_Co___Ltd_JKM410M_72HL_V"  # database identifier (kept for traceability)
MODULE_DISPLAY_NAME = "Jinko Solar Tiger Neo 48HL4M-DV 460 Wp"
MODULE_DATASHEET_URL = "https://jinkosolar.eu/wp-content/uploads/2025/05/JKM450-475N-48HL4M-DV-Z3-EU-DE.pdf"
CABLE_MATERIAL = "Cu"
J_MIN_A_CM2    = 1e-6

# ---- Module datasheet parameters (as in your code) ----
Voc_ref = 36.22   # V
Isc_ref = 15.93   # A
Vmp_ref = 30.51   # V
Imp_ref = 15.08   # A
alpha_Isc_pct_per_C = 0.045
beta_Voc_pct_per_C  = -0.25
cells_in_series = 48
alpha_sc = (alpha_Isc_pct_per_C/100.0) * Isc_ref
beta_voc = (beta_Voc_pct_per_C /100.0) * Voc_ref
gamma_pmp = -0.29

# --- Sidebar credit (above "Inputs") ---
with st.sidebar:
    st.markdown(
        """
        <div class="brand-badge">
            Built by: <a href="https://x.com/tobias_franzbrb" target="_blank">Tobias Franz</a>
        </div>
        <style>
        .brand-badge {
            font-size: 0.9rem;
            padding: 6px 10px;
            border-radius: 8px;
            border: 1px solid rgba(0,0,0,0.08);
            margin: 0.25rem 0 0.5rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================
# Debounced sidebar form: Inputs (Apply to recompute)
# =========================

# 1) One-time init of committed defaults (used for all calculations)
if "committed" not in st.session_state:
    st.session_state.committed = dict(
        site_key="Seville_ES",
        year=2023,
        sm=6, sd=25,
        wm=12, wd=25,
        SURFACE_TILT=0.0,
        SURFACE_AZIMUTH=180.0,
        N_SERIES=37,
        N_PARALLEL=63,
        SEG_LEN_M=0.8,
        ARR_CABLE_ONE_WAY_M=25.0,
        SOILING_PCT=10,
        U_NERNST_V=1.23,
        TAFEL_SLOPE_V_DEC=0.06,
        J0_A_CM2=1.5e-6,
        R_CELL_OHM_CM2=0.15,
        CELL_ACTIVE_AREA_M2=0.09,
        N_CELLS_PER_STACK=600,
        ELEC_N_SERIES_STACKS=1,
        ELEC_N_PARALLEL_STRINGS=1,
        FARADAIC_EFF_H2=1.0,
    )

# 2) Sidebar form (editing here does NOT trigger recompute until "Apply inputs")
with st.sidebar.form("inputs_form", border=True):
    st.title("Inputs")

    # --- Apply button ---
    apply_clicked = st.form_submit_button("Apply inputs", type="primary", use_container_width=True)

    # --- General ---
    st.subheader("General")
    _keys = list(LOCATIONS.keys())
    site_key_idx = _keys.index(st.session_state.committed["site_key"])
    site_key_form = st.selectbox(
        "Location", _keys, index=site_key_idx,
        format_func=lambda k: LOCATIONS[k]["label"],
        key="form_site_key"
    )

    year_form = st.slider("Year", min_value=2005, max_value=2023,
                          value=int(st.session_state.committed["year"]), step=1, key="form_year")

    def md_picker_form(label, cm, cd, year_val):
        colm, cold = st.columns([2, 1])
        month = colm.selectbox(f"{label} — Month", list(range(1, 13)),
                               index=int(cm) - 1, key=f"form_{label}_m")
        day = cold.number_input(f"{label} — Day", min_value=1, max_value=31,
                                value=int(cd), step=1, key=f"form_{label}_d")
        import calendar
        maxd = calendar.monthrange(int(year_val), int(month))[1]
        day = min(int(day), maxd)
        return int(month), int(day)

    sm_form, sd_form = md_picker_form("Summer Day",
                                      st.session_state.committed["sm"],
                                      st.session_state.committed["sd"],
                                      year_form)
    wm_form, wd_form = md_picker_form("Winter Day",
                                      st.session_state.committed["wm"],
                                      st.session_state.committed["wd"],
                                      year_form)

    # --- PV array ---
    st.subheader("PV Array")
    SURFACE_TILT_form = st.number_input("Surface tilt (deg)", min_value=0.0, max_value=90.0,
                                        value=float(st.session_state.committed["SURFACE_TILT"]), step=1.0,
                                        key="form_SURFACE_TILT")
    SURFACE_AZIMUTH_form = st.number_input("Surface azimuth (deg, pvlib conv.)", min_value=0.0, max_value=360.0,
                                           value=float(st.session_state.committed["SURFACE_AZIMUTH"]), step=1.0,
                                           key="form_SURFACE_AZIMUTH")
    N_SERIES_form = st.number_input("Modules per string (Ns)", min_value=1,
                                    value=int(st.session_state.committed["N_SERIES"]), step=1,
                                    key="form_N_SERIES")
    N_PARALLEL_form = st.number_input("Strings in parallel (Np)", min_value=1,
                                      value=int(st.session_state.committed["N_PARALLEL"]), step=1,
                                      key="form_N_PARALLEL")
    SEG_LEN_M_form = st.number_input("Inter-module jumper one-way length (m)", min_value=0.0,
                                     value=float(st.session_state.committed["SEG_LEN_M"]), step=0.1,
                                     key="form_SEG_LEN_M")
    ARR_CABLE_ONE_WAY_M_form = st.number_input("Array cable one-way length (m)", min_value=0.0,
                                               value=float(st.session_state.committed["ARR_CABLE_ONE_WAY_M"]), step=1.0,
                                               key="form_ARR_CABLE_ONE_WAY_M")
    SOILING_PCT_form = st.slider("Soiling losses (%)", min_value=0, max_value=20,
                                 value=int(st.session_state.committed["SOILING_PCT"]), step=1,
                                 key="form_SOILING_PCT")

    # --- Electrolyzer ---
    st.subheader("Electrolyzer Array")
    U_NERNST_V_form = st.number_input("U_Nernst (V per cell)", min_value=0.0,
                                      value=float(st.session_state.committed["U_NERNST_V"]), step=0.01,
                                      key="form_U_NERNST_V")
    TAFEL_SLOPE_V_DEC_form = st.number_input("Tafel slope (V/decade)", min_value=0.0,
                                             value=float(st.session_state.committed["TAFEL_SLOPE_V_DEC"]),
                                             step=0.005, format="%.3f",
                                             key="form_TAFEL_SLOPE_V_DEC")
    J0_A_CM2_form = st.number_input("Apparent exchange current density (A/cm²)", min_value=0.0,
                                    value=float(st.session_state.committed["J0_A_CM2"]), step=1e-6, format="%.6f",
                                    key="form_J0_A_CM2")
    R_CELL_OHM_CM2_form = st.number_input("Ohmic cell resistance (Ω·cm²)", min_value=0.0,
                                          value=float(st.session_state.committed["R_CELL_OHM_CM2"]), step=0.01,
                                          key="form_R_CELL_OHM_CM2")
    CELL_ACTIVE_AREA_M2_form = st.number_input("Cell active area (m²)", min_value=0.0001,
                                               value=float(st.session_state.committed["CELL_ACTIVE_AREA_M2"]),
                                               step=0.005, key="form_CELL_ACTIVE_AREA_M2")
    N_CELLS_PER_STACK_form = st.number_input("Number of cells per stack", min_value=1,
                                             value=int(st.session_state.committed["N_CELLS_PER_STACK"]), step=10,
                                             key="form_N_CELLS_PER_STACK")
    ELEC_N_SERIES_STACKS_form = st.number_input("Stacks in series per string", min_value=1,
                                                value=int(st.session_state.committed["ELEC_N_SERIES_STACKS"]), step=1,
                                                key="form_ELEC_N_SERIES_STACKS")
    ELEC_N_PARALLEL_STRINGS_form = st.number_input("Electrolyzer strings in parallel", min_value=1,
                                                   value=int(st.session_state.committed["ELEC_N_PARALLEL_STRINGS"]),
                                                   step=1, key="form_ELEC_N_PARALLEL_STRINGS")
    FARADAIC_EFF_H2_form = st.number_input("Faradaic efficiency to H₂ (0..1)", min_value=0.0, max_value=1.0,
                                           value=float(st.session_state.committed["FARADAIC_EFF_H2"]), step=0.01,
                                           key="form_FARADAIC_EFF_H2")


# 3) When Apply is pressed, copy staged → committed (single rerun)
if apply_clicked:
    st.session_state.committed = dict(
        site_key=site_key_form,
        year=int(year_form),
        sm=int(sm_form), sd=int(sd_form),
        wm=int(wm_form), wd=int(wd_form),
        SURFACE_TILT=float(SURFACE_TILT_form),
        SURFACE_AZIMUTH=float(SURFACE_AZIMUTH_form),
        N_SERIES=int(N_SERIES_form),
        N_PARALLEL=int(N_PARALLEL_form),
        SEG_LEN_M=float(SEG_LEN_M_form),
        ARR_CABLE_ONE_WAY_M=float(ARR_CABLE_ONE_WAY_M_form),
        SOILING_PCT=int(SOILING_PCT_form),
        U_NERNST_V=float(U_NERNST_V_form),
        TAFEL_SLOPE_V_DEC=float(TAFEL_SLOPE_V_DEC_form),
        J0_A_CM2=float(J0_A_CM2_form),
        R_CELL_OHM_CM2=float(R_CELL_OHM_CM2_form),
        CELL_ACTIVE_AREA_M2=float(CELL_ACTIVE_AREA_M2_form),
        N_CELLS_PER_STACK=int(N_CELLS_PER_STACK_form),
        ELEC_N_SERIES_STACKS=int(ELEC_N_SERIES_STACKS_form),
        ELEC_N_PARALLEL_STRINGS=int(ELEC_N_PARALLEL_STRINGS_form),
        FARADAIC_EFF_H2=float(FARADAIC_EFF_H2_form),
    )

# 4) Use ONLY committed values below this line
site_key = st.session_state.committed["site_key"]
year = st.session_state.committed["year"]
sm = st.session_state.committed["sm"]; sd = st.session_state.committed["sd"]
wm = st.session_state.committed["wm"]; wd = st.session_state.committed["wd"]

SURFACE_TILT    = st.session_state.committed["SURFACE_TILT"]
SURFACE_AZIMUTH = st.session_state.committed["SURFACE_AZIMUTH"]
N_SERIES        = st.session_state.committed["N_SERIES"]
N_PARALLEL      = st.session_state.committed["N_PARALLEL"]
SEG_LEN_M       = st.session_state.committed["SEG_LEN_M"]
ARR_CABLE_ONE_WAY_M = st.session_state.committed["ARR_CABLE_ONE_WAY_M"]
SOILING_PCT     = st.session_state.committed["SOILING_PCT"]

U_NERNST_V        = st.session_state.committed["U_NERNST_V"]
TAFEL_SLOPE_V_DEC = st.session_state.committed["TAFEL_SLOPE_V_DEC"]
J0_A_CM2          = st.session_state.committed["J0_A_CM2"]
R_CELL_OHM_CM2    = st.session_state.committed["R_CELL_OHM_CM2"]
CELL_ACTIVE_AREA_M2     = st.session_state.committed["CELL_ACTIVE_AREA_M2"]
N_CELLS_PER_STACK       = st.session_state.committed["N_CELLS_PER_STACK"]
ELEC_N_SERIES_STACKS    = st.session_state.committed["ELEC_N_SERIES_STACKS"]
ELEC_N_PARALLEL_STRINGS = st.session_state.committed["ELEC_N_PARALLEL_STRINGS"]
FARADAIC_EFF_H2         = st.session_state.committed["FARADAIC_EFF_H2"]

# Build day strings (from committed values)
WINTER_DAY = f"{year}-{wm:02d}-{wd:02d}"
SUMMER_DAY = f"{year}-{sm:02d}-{sd:02d}"

# Gather site config as before
cfg = LOCATIONS[site_key]
SITE_LAT = cfg["lat"]; SITE_LON = cfg["lon"]; SITE_ALT = cfg["alt"]; SITE_TZ = cfg["tz"]
SITE_NAME = cfg.get("label", site_key)


# Build day strings
WINTER_DAY = f"{year}-{wm:02d}-{wd:02d}"
SUMMER_DAY = f"{year}-{sm:02d}-{sd:02d}"

# Gather site config
cfg = LOCATIONS[site_key]
SITE_LAT = cfg["lat"]; SITE_LON = cfg["lon"]; SITE_ALT = cfg["alt"]; SITE_TZ = cfg["tz"]
SITE_NAME = cfg.get("label", site_key)

# Helpful header
st.title("PV Array → direct DC coupling → Electrolyzer Array")
st.caption(f"Site: **{SITE_NAME}** · Year: **{year}** · Summer Day: **{SUMMER_DAY}** · Winter Day: **{WINTER_DAY}**")

# =========================
# Cache: PVGIS fetch + effective irradiance (pre-soiling)
# =========================
@st.cache_data(show_spinner=True, ttl=24*3600)
def fetch_pvgis_and_effective_irradiance(site_lat, site_lon, site_alt, site_tz, start_year, surface_tilt, surface_azimuth):
    START = start_year; END = start_year
    SITE_LAT_local, SITE_LON_local, SITE_ALT_local, SITE_TZ_local = site_lat, site_lon, site_alt, site_tz
    SURFACE_TILT_local, SURFACE_AZIMUTH_local = surface_tilt, surface_azimuth

    global data, meta, aoi
    SITE_LAT = SITE_LAT_local; SITE_LON = SITE_LON_local; SITE_ALT = SITE_ALT_local; SITE_TZ = SITE_TZ_local
    SURFACE_TILT = SURFACE_TILT_local; SURFACE_AZIMUTH = SURFACE_AZIMUTH_local

    data, meta = iotools.get_pvgis_hourly(
        latitude=SITE_LAT, longitude=SITE_LON,
        surface_tilt=SURFACE_TILT, surface_azimuth=SURFACE_AZIMUTH,
        start=START, end=END,
        map_variables=True, components=True, usehorizon=True, url="https://re.jrc.ec.europa.eu/api/v5_3/"
    )
    if "time" in data.columns:
        data = data.set_index("time")
    if data.index.tz is None:
        data = data.tz_localize("UTC")
    data = data.tz_convert(SITE_TZ).sort_index()
    data["poa_global"] = data["poa_direct"] + data["poa_sky_diffuse"] + data["poa_ground_diffuse"]

    loc = location.Location(SITE_LAT, SITE_LON, tz=SITE_TZ, altitude=SITE_ALT)
    sp  = loc.get_solarposition(data.index)
    aoi = irradiance.aoi(SURFACE_TILT, SURFACE_AZIMUTH, sp['apparent_zenith'], sp['azimuth'])

    b0 = 0.04
    iam_beam = iam.ashrae(aoi, b=b0)
    diffuse = iam.marion_diffuse('ashrae', surface_tilt=SURFACE_TILT, b=b0)
    iam_sky, iam_ground = diffuse['sky'], diffuse['ground']

    E_eff_base = (data['poa_direct']*iam_beam + data['poa_sky_diffuse']*iam_sky + data['poa_ground_diffuse']*iam_ground).clip(lower=0).fillna(0.0)
    return data, meta, E_eff_base

# Fetch (cached unless site/year/tilt/azimuth change)
with st.spinner("Fetching PVGIS data and computing effective irradiance..."):
    data, meta, E_eff_base = fetch_pvgis_and_effective_irradiance(
        SITE_LAT, SITE_LON, SITE_ALT, SITE_TZ, year, SURFACE_TILT, SURFACE_AZIMUTH
    )

# Apply soiling losses to effective irradiance
E_eff = E_eff_base * (1.0 - SOILING_PCT / 100.0)

# =========================
# 3) Cell temperature (SAPM)
# =========================
sapm = temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_glass"]
data["cell_temperature"] = temperature.sapm_cell(
    poa_global=data["poa_global"],
    temp_air=data["temp_air"],
    wind_speed=data["wind_speed"],
    **sapm
)

# =========================
# 4) Module IV parameters (CEC/DeSoto) & summaries
# =========================
I_L_ref, I_o_ref, R_s, R_sh_ref, a_ref, Adjust = fit_cec_sam(
    celltype='monoSi',
    v_mp=Vmp_ref, i_mp=Imp_ref, v_oc=Voc_ref, i_sc=Isc_ref,
    alpha_sc=alpha_sc, beta_voc=beta_voc, gamma_pmp=gamma_pmp,
    cells_in_series=cells_in_series, temp_ref=25
)

IL, I0, Rs, Rsh, nNsVth = pvsystem.calcparams_cec(
    effective_irradiance=E_eff,
    temp_cell=data["cell_temperature"],
    alpha_sc=alpha_sc,
    a_ref=a_ref, I_L_ref=I_L_ref, I_o_ref=I_o_ref, R_sh_ref=R_sh_ref, R_s=R_s,
    Adjust=Adjust,
    EgRef=1.121, dEgdT=-0.0002677
)

curve_info = pvsystem.singlediode(
    photocurrent=IL, saturation_current=I0,
    resistance_series=Rs, resistance_shunt=Rsh, nNsVth=nNsVth,
    method="lambertw"
)

# =========================
# 5) DC ohmic resistances & helpers
# =========================
def _rho_T(material: str, T_C: float) -> float:
    if material.lower() == "al":
        rho20, alpha = 2.82e-8, 0.0039
    else:
        rho20, alpha = 1.724e-8, 0.00393
    return rho20 * (1.0 + alpha * (T_C - 20.0))

def loop_R(one_way_m: float, N_parallel: int, *, module_imax_A: float = 16.0, design_factor: float = 1.25,
           j_cu_A_per_mm2: float = 4.0, T_C: float = 45.0, standard_sizes: list[float] | None = None):
    sizes = standard_sizes or [4, 6, 10, 16, 25, 35, 50, 70, 95, 120, 150, 185, 240, 300, 400, 500, 630, 800, 1000]
    rho20 = 1.724e-8; alpha = 0.00393
    rho = rho20 * (1.0 + alpha * (T_C - 20.0))
    I_array_max = float(N_parallel) * float(module_imax_A)
    I_design = I_array_max * float(design_factor)
    A_required = I_design / max(j_cu_A_per_mm2, 1e-9)   # mm²
    A_sel = next((s for s in sizes if s >= A_required - 1e-12), sizes[-1])
    A_m2 = A_sel * 1e-6
    R_loop = rho * (2.0 * one_way_m) / A_m2
    return float(R_loop), float(A_sel), float(A_required)

def string_wiring_R(Ns: int, seg_len_m: float, A_mm2: float, material="Cu", T_C=45.0) -> float:
    if Ns <= 1:
        return 0.0
    rho = _rho_T(material, T_C)
    A_m2 = A_mm2 * 1e-6
    total_one_way = (Ns - 1) * seg_len_m
    return rho * (2.0 * total_one_way) / A_m2

R_string = string_wiring_R(N_SERIES, SEG_LEN_M, CROSS_SEC_MM2, material=CABLE_MATERIAL, T_C=CABLE_TEMP_C)
R_array, ARR_CABLE_CROSS_SEC_MM2, ARR_CABLE_CSA_REQUIRED = loop_R(
    one_way_m=ARR_CABLE_ONE_WAY_M, N_parallel=N_PARALLEL, module_imax_A=16.0,
    design_factor=1.25, j_cu_A_per_mm2=4.0, T_C=ARR_CABLE_TEMP_C
)

info_lines = [
    f"String jumper R: {R_string:.5f} Ω per string",
    f"Array cable R: {R_array:.5f} Ω at array level",
    f"Array cable CSA selected: {ARR_CABLE_CROSS_SEC_MM2:.1f} mm² (required: {ARR_CABLE_CSA_REQUIRED:.1f} mm²)",
]

# =========================
# 6) Electrolyzer: cell → stack → string → array
# =========================
A_cell_cm2 = CELL_ACTIVE_AREA_M2 * 1e4

def u_cell_from_j(j_A_cm2: np.ndarray) -> np.ndarray:
    j = np.maximum(j_A_cm2, J_MIN_A_CM2)
    eta_kin = TAFEL_SLOPE_V_DEC * np.log10(j / J0_A_CM2)
    eta_ohm = j * R_CELL_OHM_CM2
    return U_NERNST_V + eta_kin + eta_ohm

def v_elec_array_from_Iarray(I_array_A: np.ndarray) -> np.ndarray:
    I_per_stack = I_array_A / max(ELEC_N_PARALLEL_STRINGS, 1)
    j = I_per_stack / max(A_cell_cm2, 1e-9)
    U_cell  = u_cell_from_j(j)
    U_stack = N_CELLS_PER_STACK * U_cell
    U_str   = ELEC_N_SERIES_STACKS * U_stack
    return U_str

# =========================
# 7) Coupling PV ↔ Electrolyzer (hourly)
# =========================
def array_iv_at_time(t, Ns, Np, Rstr, Rarr, IL_ser, I0_ser, Rs_ser, Rsh_ser, nNsVth_ser, iv_points=220):
    try:
        Voc_mod = float(curve_info.at[t, "v_oc"])
        Isc_mod = float(curve_info.at[t, "i_sc"])
    except Exception:
        return None
    if not np.isfinite(Voc_mod) or not np.isfinite(Isc_mod) or Isc_mod <= 0:
        return None

    IL_t = float(IL_ser.loc[t]); I0_t = float(I0_ser.loc[t])
    Rs_t = float(Rs_ser.loc[t]); Rsh_t = float(Rsh_ser.loc[t]); nVth_t = float(nNsVth_ser.loc[t])

    I_mod = np.linspace(0.0, Isc_mod, iv_points)
    V_mod = pvsystem.v_from_i(photocurrent=IL_t, saturation_current=I0_t,
                              resistance_series=Rs_t, resistance_shunt=Rsh_t,
                              nNsVth=nVth_t, current=I_mod, method="lambertw")

    V_str = Ns * V_mod - I_mod * Rstr
    I_arr = Np * I_mod
    V_arr = V_str - I_arr * Rarr

    m = np.isfinite(V_arr) & np.isfinite(I_arr)
    if not np.any(m):
        return None
    V_arr = V_arr[m]; I_arr = I_arr[m]

    idx_pos = np.where(V_arr >= 0)[0]
    if idx_pos.size == 0:
        return np.array([0.0]), np.array([0.0]), np.array([0.0]), 0.0, 0.0, 0.0

    k_last = idx_pos[-1]
    if k_last < len(V_arr) - 1 and V_arr[k_last] > 0 and V_arr[k_last + 1] < 0:
        V0, V1 = V_arr[k_last], V_arr[k_last + 1]
        I0p, I1p = I_arr[k_last], I_arr[k_last + 1]
        I_sc_eff = I0p - V0 * (I1p - I0p) / (V1 - V0)
        V_arr = np.concatenate([V_arr[:k_last + 1], [0.0]])
        I_arr = np.concatenate([I_arr[:k_last + 1], [float(I_sc_eff)]])
    else:
        V_arr[k_last] = max(0.0, V_arr[k_last])

    keep = V_arr >= 0
    V_arr = V_arr[keep]; I_arr = I_arr[keep]
    P_arr = V_arr * I_arr
    if P_arr.size == 0:
        return None
    i_max = int(np.nanargmax(P_arr))
    return V_arr, I_arr, P_arr, float(V_arr[i_max]), float(I_arr[i_max]), float(P_arr[i_max])

def find_coupling_point_at_time(t, Ns, Np, Rstr, Rarr, iv_points=240):
    res = array_iv_at_time(t, Ns, Np, Rstr, Rarr, IL, I0, Rs, Rsh, nNsVth, iv_points=iv_points)
    if res is None:
        return 0.0, 0.0, 0.0, False
    V_pv, I_pv, P_pv, *_ = res
    if len(I_pv) < 2:
        return 0.0, 0.0, 0.0, False

    V_el = v_elec_array_from_Iarray(I_pv)
    delta = V_pv - V_el
    valid = np.isfinite(delta)
    Iv, Vv, dv = I_pv[valid], V_pv[valid], delta[valid]
    if len(Iv) < 2:
        return 0.0, 0.0, 0.0, False

    crossings = np.where(np.diff(np.sign(dv)) != 0)[0]
    if len(crossings) >= 1:
        cand = []
        for k0 in crossings:
            Iv0, Iv1 = Iv[k0], Iv[k0+1]
            d0, d1 = dv[k0], dv[k0+1]
            I_op = Iv0 if (d1 - d0) == 0 else Iv0 - d0 * (Iv1 - Iv0) / (d1 - d0)
            I_op = float(np.clip(I_op, min(Iv0, Iv1), max(Iv0, Iv1)))
            V_op = float(np.interp(I_op, I_pv, V_pv))
            P_op = I_op * V_op
            cand.append((P_op, I_op, V_op))
        P_op, I_op, V_op = max(cand, key=lambda x: x[0])
        return I_op, V_op, P_op, True
    else:
        k = int(np.nanargmin(np.abs(dv)))
        I_op = float(Iv[k]); V_op = float(Vv[k])
        return I_op, V_op, I_op * V_op, False

# Build yearly coupling time series
op_times, I_op_list, V_op_list, P_op_list = [], [], [], []
for t in curve_info.index:
    I_op, V_op, P_op, _ = find_coupling_point_at_time(t, N_SERIES, N_PARALLEL, R_string, R_array)
    op_times.append(t); I_op_list.append(I_op); V_op_list.append(V_op); P_op_list.append(P_op)

coupling_results = pd.DataFrame(
    {"I_op_A": I_op_list, "V_op_V": V_op_list, "P_op_W": P_op_list},
    index=pd.to_datetime(op_times)
).sort_index().fillna(0.0)

# Electrolyzer derived quantities
j_op = (coupling_results["I_op_A"] / max(ELEC_N_PARALLEL_STRINGS, 1)) / max(A_cell_cm2, 1e-9)
U_cell_op  = u_cell_from_j(j_op.values)
U_stack_op = N_CELLS_PER_STACK * U_cell_op
U_string_op= ELEC_N_SERIES_STACKS * U_stack_op
coupling_results["j_cell_Acm2"] = j_op.values
coupling_results["U_cell_V"]    = U_cell_op
coupling_results["U_stack_V"]   = U_stack_op
coupling_results["U_string_V"]  = U_string_op

# PV side breakdown
idx = coupling_results.index
I_array = coupling_results["I_op_A"].reindex(idx).astype(float)
I_module = I_array / max(N_PARALLEL, 1)
I_string = I_module.copy()

V_module = pd.Series(0.0, index=idx, dtype=float)
mask = I_module > 0
if mask.any():
    V_mod_vals = pvsystem.v_from_i(
        photocurrent=IL.reindex(idx).astype(float).values[mask.values],
        saturation_current=I0.reindex(idx).astype(float).values[mask.values],
        resistance_series=Rs.reindex(idx).astype(float).values[mask.values],
        resistance_shunt=Rsh.reindex(idx).astype(float).values[mask.values],
        nNsVth=nNsVth.reindex(idx).astype(float).values[mask.values],
        current=I_module[mask].values,
        method="lambertw"
    )
    V_module.loc[mask] = np.where(np.isfinite(V_mod_vals), V_mod_vals, 0.0)

V_string = (N_SERIES * V_module - I_module * R_string).clip(lower=0.0)
V_array  = (V_string - I_array * R_array).clip(lower=0.0)

# Save PV variables
coupling_results["Single_PV_module_voltage"] = V_module.values
coupling_results["Single_PV_module_current"] = I_module.values
coupling_results["Single_PV_module_power"]   = (V_module * I_module).values
coupling_results["PV_string_current"] = I_string.values
coupling_results["PV_string_voltage"] = V_string.values
coupling_results["PV_string_power"]   = (V_string * I_string).values
coupling_results["PV_array_voltage"] = V_array.values
coupling_results["PV_array_current"] = I_array.values
coupling_results["PV_array_power"]   = (V_array * I_array).values

# Electrolyzer side breakdown
I_stack = (coupling_results["j_cell_Acm2"] * A_cell_cm2).astype(float)
coupling_results["Single_Stack_voltage"] = coupling_results["U_stack_V"].astype(float).values
coupling_results["Single_Stack_current"] = I_stack.values
coupling_results["Single_Stack_Power"]   = (coupling_results["Single_Stack_voltage"] * I_stack).values
coupling_results["Electrolyzer_string_voltage"]  = coupling_results["U_string_V"].astype(float).values
coupling_results["Elecgtrolyzer_string_current"] = I_stack.values
coupling_results["Electrolyzer_string_power"]    = (coupling_results["Electrolyzer_string_voltage"] * I_stack).values
coupling_results["Electrolyzer_array_voltage"] = coupling_results["U_string_V"].astype(float).values
coupling_results["Electrolyzer_array_current"] = coupling_results["I_op_A"].astype(float).values
coupling_results["Electrolyzer_array_power"]   = (
    coupling_results["Electrolyzer_array_voltage"] * coupling_results["Electrolyzer_array_current"]
).values

# Maxima dict
_added_cols = [
    "Single_PV_module_voltage", "Single_PV_module_current", "Single_PV_module_power",
    "PV_string_current", "PV_string_voltage", "PV_string_power",
    "PV_array_voltage", "PV_array_current", "PV_array_power",
    "Single_Stack_voltage", "Single_Stack_current", "Single_Stack_Power",
    "Electrolyzer_string_voltage", "Elecgtrolyzer_string_current", "Electrolyzer_string_power",
    "Electrolyzer_array_voltage", "Electrolyzer_array_current", "Electrolyzer_array_power",
]
maxima = {col: float(np.nanmax(coupling_results[col].values)) if col in coupling_results.columns else np.nan
          for col in _added_cols}

# Energy: PV @ MPP baseline (with DC losses) vs coupled
IV_POINTS_ENERGY = 220
def compute_array_mpp_series(Ns, Np, Rstr, Rarr, iv_points=IV_POINTS_ENERGY):
    ts, pmp = [], []
    for t in curve_info.index:
        res = array_iv_at_time(t, Ns, Np, Rstr, Rarr, IL, I0, Rs, Rsh, nNsVth, iv_points=iv_points)
        if res is None:
            ts.append(t); pmp.append(np.nan)
        else:
            ts.append(t); pmp.append(res[-1])
    s = pd.Series(pmp, index=pd.to_datetime(ts)).sort_index()
    return s.reindex(data.index.sort_values()).fillna(0.0)

Pmp_fullR_W = compute_array_mpp_series(N_SERIES, N_PARALLEL, R_string, R_array)
E_mpp_fullR_kWh = Pmp_fullR_W.sum() / 1000.0
E_coupled_kWh = coupling_results["P_op_W"].sum() / 1000.0

# Hydrogen production (Faraday)
F = 96485.3329                 # C/mol e-
M_H2_KG_PER_MOL = 0.002016     # kg/mol H2
ts = coupling_results.index
dt_s = (ts.to_series().shift(-1) - ts.to_series()).dt.total_seconds()
dt_s.iloc[-1] = dt_s.median() if np.isfinite(dt_s.median()) else 3600.0
dt_s = dt_s.values

I_arr_for_H2 = coupling_results["j_cell_Acm2"].values * A_cell_cm2 * N_CELLS_PER_STACK * (
    ELEC_N_SERIES_STACKS * ELEC_N_PARALLEL_STRINGS
)
mol_H2 = (I_arr_for_H2 * dt_s) / (2.0 * F) * FARADAIC_EFF_H2
kg_H2_interval = mol_H2 * M_H2_KG_PER_MOL
kg_H2_total = np.nansum(kg_H2_interval)

max_current_density = float(np.nanmax(coupling_results["j_cell_Acm2"].values)) if len(coupling_results) else np.nan
_min_mask = coupling_results["j_cell_Acm2"].values > 0
min_current_density = float(np.nanmin(coupling_results["j_cell_Acm2"].values[_min_mask])) if np.any(_min_mask) else 0.0

SEC_kWh_per_kg = (E_coupled_kWh / kg_H2_total) if kg_H2_total > 0 else np.nan
N_cells_total = N_CELLS_PER_STACK * (ELEC_N_SERIES_STACKS * ELEC_N_PARALLEL_STRINGS)
A_cell_total = A_cell_cm2 * N_cells_total
Max_stack_power = float(np.nanmax(coupling_results["j_cell_Acm2"].values*A_cell_cm2*coupling_results["U_stack_V"].values))
Max_stack_Voltage = float(np.nanmax(coupling_results["U_stack_V"].values))
Max_stack_Current = float(np.nanmax(coupling_results["j_cell_Acm2"].values*A_cell_cm2))
Max_electrolyzer_array_power = float(np.nanmax(coupling_results["I_op_A"].values*coupling_results["V_op_V"].values))
util_pct = (100.0*E_coupled_kWh/E_mpp_fullR_kWh) if E_mpp_fullR_kWh > 0 else np.nan

# ======================================================
# PLOTLY HELPERS — Interactive figures
# ======================================================
def _hourly_times_for_day(index, day_str):
    m = index.strftime("%Y-%m-%d") == day_str
    return index[m]

def _iv_curves_for_day(day_str):
    """Return a list of dicts per hour: {'h', 'V_arr', 'I_arr', 'Vmp', 'Imp'} for the chosen day."""
    times = _hourly_times_for_day(curve_info.index, day_str)
    out = []
    for t in times:
        res = array_iv_at_time(t, N_SERIES, N_PARALLEL, R_string, R_array,
                               IL, I0, Rs, Rsh, nNsVth, iv_points=220)
        if res is None:
            continue
        V_arr, I_arr, P_arr, Vmp, Imp, Pmp = res
        out.append({"h": int(pd.to_datetime(t).hour),
                    "V_arr": V_arr, "I_arr": I_arr, "Vmp": Vmp, "Imp": Imp})
    out = sorted(out, key=lambda d: d["h"])
    return out

def _electrolyzer_curve_for_day(day_str):
    """Build electrolyzer I–V curve span for the max I seen that day."""
    times = _hourly_times_for_day(curve_info.index, day_str)
    Imax = 0.0
    for t in times:
        res = array_iv_at_time(t, N_SERIES, N_PARALLEL, R_string, R_array,
                               IL, I0, Rs, Rsh, nNsVth, iv_points=240)
        if res is None:
            continue
        _, I_arr, _, *_ = res
        if len(I_arr):
            Imax = max(Imax, float(np.nanmax(I_arr)))
    if Imax <= 0:
        return np.array([]), np.array([])
    I_grid = np.linspace(0.0, 1.05*Imax, 300)
    V_el   = v_elec_array_from_Iarray(I_grid)
    return V_el, I_grid

def _coupling_points_for_day(day_str):
    times = _hourly_times_for_day(coupling_results.index, day_str)
    return coupling_results.loc[times, ["V_op_V", "I_op_A"]].dropna()

def _power_series_for_day(day_str):
    times = _hourly_times_for_day(curve_info.index, day_str)
    pmp = Pmp_fullR_W.reindex(times).fillna(0.0)
    pcpl = coupling_results.loc[times, "P_op_W"].fillna(0.0)
    return pmp, pcpl

# --------- FIGURE SET A: IV (Winter+Summer) ----------
def _mk_iv_figure_pair(winter_day, summer_day):
    import numpy as np
    MPP_MARKER = "circle"
    LINESTYLES = ["solid", "dash", "dashdot", "dot"]
    BLUES   = ['#2c7bb6', '#3a89c9', '#4f9bd4', '#66addf']
    GREENS  = ['#1a9850', '#4daf4a', '#66bd63', '#99d594']
    ORANGES = ['#fdae61', '#f98e52', '#f46d43', '#f2703d']
    REDS    = ['#d73027', '#d7191c']

    def _class_color(i_class: str, counters: dict) -> str:
        if i_class == 'blue':
            c = BLUES[counters['blue'] % len(BLUES)];   counters['blue']  += 1; return c
        if i_class == 'green':
            c = GREENS[counters['green'] % len(GREENS)]; counters['green'] += 1; return c
        if i_class == 'orange':
            c = ORANGES[counters['orange'] % len(ORANGES)]; counters['orange'] += 1; return c
        c = REDS[counters['red'] % len(REDS)]; counters['red'] += 1; return c

    def _class_from_Imax(Imax_val, q1, q2, q3):
        if Imax_val <= q1:      return 'blue'
        elif Imax_val <= q2:    return 'green'
        elif Imax_val <= q3:    return 'orange'
        else:                   return 'red'

    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=False,
        subplot_titles=(
            f"Array I-U — {winter_day} (Winter)  |  Ns={N_SERIES}, Np={N_PARALLEL}",
            f"Array I-U — {summer_day} (Summer)  |  Ns={N_SERIES}, Np={N_PARALLEL}",
        ),
        horizontal_spacing=0.08
    )

    def _hourly_times_for_day(index, day_str):
        m = index.strftime("%Y-%m-%d") == day_str
        return index[m]

    def _add_iv_panel(day_str, c):
        """Draw one day panel; set X range from 1.1× max(PV V, EL V). Return day's max Imp."""
        times_day = _hourly_times_for_day(curve_info.index, day_str)
        hour_Imax = {}
        iv_cache  = {}
        imax_mpp  = 0.0

        # Track maxima for X-axis
        vmax_pv_curves = 0.0
        vmax_el_curve  = 0.0

        for t in times_day:
            res = array_iv_at_time(t, N_SERIES, N_PARALLEL, R_string, R_array,
                                   IL, I0, Rs, Rsh, nNsVth, iv_points=220)
            if res is None:
                continue
            V_arr, I_arr, P_arr, Vmp, Imp, Pmp = res
            iv_cache[t] = (V_arr, I_arr, Vmp, Imp)
            h = int(pd.to_datetime(t).hour)
            if len(I_arr):
                hour_Imax[h] = max(hour_Imax.get(h, 0.0), float(np.nanmax(I_arr)))
            if len(V_arr):
                vmax_pv_curves = max(vmax_pv_curves, float(np.nanmax(V_arr)))
            if np.isfinite(Imp):
                imax_mpp = max(imax_mpp, float(Imp))

        if not hour_Imax:
            fig.add_annotation(text=f"No sun on {day_str}",
                               xref=f"x{c}", yref=f"y{c}",
                               showarrow=False, x=0.5, y=0.5)
            fig.update_xaxes(title_text="Array Voltage U (V)", row=1, col=c, range=[0.0, 1.0])
            # Y will be set globally later; keep autorange here
            fig.update_yaxes(title_text="Array Current I (A)", row=1, col=c, autorange=True)
            return 0.0

        vals = np.array(list(hour_Imax.values()))
        q1, q2, q3 = np.quantile(vals, [0.25, 0.50, 0.75])
        class_counters = {'blue':0, 'green':0, 'orange':0, 'red':0}

        for t in sorted(iv_cache.keys()):
            V_arr, I_arr, Vmp, Imp = iv_cache[t]
            h = int(pd.to_datetime(t).hour)
            i_class  = _class_from_Imax(hour_Imax.get(h, 0.0), q1, q2, q3)
            color    = _class_color(i_class, class_counters)
            dash     = LINESTYLES[h % len(LINESTYLES)]
            label    = f"{h:02d}:00"

            fig.add_trace(
                go.Scatter(
                    x=V_arr, y=I_arr, mode="lines",
                    name=label, legendgroup=f"hours_{day_str}",
                    line=dict(width=1.9, color=color, dash=dash),
                    hovertemplate="U=%{x:.1f} V<br>I=%{y:.1f} A<extra></extra>",
                    showlegend=True
                ),
                row=1, col=c
            )
            fig.add_trace(
                go.Scatter(
                    x=[Vmp], y=[Imp], mode="markers",
                    name="PV MPP", legendgroup=f"mpp_{day_str}",
                    showlegend=False,
                    marker=dict(size=7, symbol=MPP_MARKER, color=color,
                                line=dict(width=0.8, color="black")),
                    hovertemplate="Vmp=%{x:.1f} V<br>Imp=%{y:.1f} A<extra></extra>"
                ),
                row=1, col=c
            )

        # Electrolyzer curve + coupling points
        Imax_day = max(hour_Imax.values()) if hour_Imax else 0.0
        if Imax_day > 0:
            I_grid = np.linspace(0.0, 1.05*Imax_day, 300)
            V_el   = v_elec_array_from_Iarray(I_grid)
            if np.size(V_el):
                vmax_el_curve = max(vmax_el_curve, float(np.nanmax(V_el)))

            fig.add_trace(
                go.Scatter(
                    x=V_el, y=I_grid, mode="lines",
                    name="Electrolyzer array", legendgroup=f"el_{day_str}",
                    line=dict(width=2.2, color="red"),
                    hovertemplate="U=%{x:.1f} V<br>I=%{y:.1f} A<extra></extra>",
                    showlegend=True
                ),
                row=1, col=c
            )
            cp = _coupling_points_for_day(day_str)
            if not cp.empty:
                fig.add_trace(
                    go.Scatter(
                        x=cp["V_op_V"], y=cp["I_op_A"], mode="markers",
                        name="Coupling points", legendgroup=f"cp_{day_str}",
                        marker=dict(symbol="x", size=7, color="red"),
                        hovertemplate="U=%{x:.1f} V<br>I=%{y:.1f} A<extra></extra>",
                        showlegend=True
                    ),
                    row=1, col=c
                )

        # Fixed X limit per your rule
        x_max = 1.1 * max(vmax_pv_curves, vmax_el_curve, 1e-9)
        fig.update_xaxes(title_text="Array Voltage U (V)", row=1, col=c, range=[0.0, x_max])
        # Y range set globally after both panels are drawn
        return imax_mpp

    imax_mpp_w = _add_iv_panel(winter_day, 1)
    imax_mpp_s = _add_iv_panel(summer_day, 2)

    # Common Y limit on BOTH subplots = 1.1 × max(Imp over the two days)
    imax_common = 1.1 * max(imax_mpp_w, imax_mpp_s, 1e-9)

    for c in (1, 2):
        fig.update_xaxes(
            showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.08)",
            zeroline=False,
            row=1, col=c
        )

    fig.update_yaxes(
        title_text="Array Current I (A)",
        range=[0.0, imax_common],
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="Array Current I (A)",
        range=[0.0, imax_common],
        row=1, col=2
    )


    fig.update_layout(
        height=560,
        plot_bgcolor="white",
        hovermode="closest",
        margin=dict(t=90, r=320, b=60, l=70),
        legend=dict(
            orientation="v",
            yanchor="top", y=1.0,
            xanchor="left", x=1.02,
            tracegroupgap=6,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.15)",
            borderwidth=1,
            title="Hour / Curves"
        ),
    )

    # Proxy legend entry for PV MPP marker
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="markers",
            name="PV MPP",
            marker=dict(size=7, symbol=MPP_MARKER, color="white",
                        line=dict(width=0.8, color="black"))
        ),
        row=1, col=1
    )
    return fig






# --------- FIGURE SET B: POWER (Winter+Summer) ONLY ----------
def _mk_power_figure_pair(winter_day, summer_day):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Powers vs Time — {winter_day}",
            f"Powers vs Time — {summer_day}",
        ),
        horizontal_spacing=0.08
    )

    # --- Compute common Y max in kW: 1.1x max PV array MPP across the two days ---
    def _times_for_day(day_str):
        m = curve_info.index.strftime("%Y-%m-%d") == day_str
        return curve_info.index[m]

    pmp_w = Pmp_fullR_W.reindex(_times_for_day(winter_day)).fillna(0.0)  # W
    pmp_s = Pmp_fullR_W.reindex(_times_for_day(summer_day)).fillna(0.0)  # W
    p_max_w = float(pmp_w.max() if not pmp_w.empty else 0.0)
    p_max_s = float(pmp_s.max() if not pmp_s.empty else 0.0)
    p_max_kw = max(p_max_w, p_max_s) / 1000.0
    y_max_kw = 1.1 * p_max_kw if p_max_kw > 0 else None

    def _add_power_panel(day_str, c):
        pmp_watts, pcpl_watts = _power_series_for_day(day_str)  # both in W
        pmp_kw = (pmp_watts / 1000.0).astype(float)
        pcpl_kw = (pcpl_watts / 1000.0).astype(float)

        fig.add_trace(
            go.Scatter(
                x=pmp_kw.index, y=pmp_kw.values, mode="lines+markers",
                name="PV @ MPP (with DC R)",
                line=dict(width=2), marker=dict(size=6, color="blue"),
                hovertemplate="Time=%{x|%H:%M}<br>P=%{y:.2f} kW<extra></extra>"
            ),
            row=1, col=c
        )
        fig.add_trace(
            go.Scatter(
                x=pcpl_kw.index, y=pcpl_kw.values, mode="markers",
                name="Coupled power",
                marker=dict(symbol="x", size=8, color="red"),
                hovertemplate="Time=%{x|%H:%M}<br>P=%{y:.2f} kW<extra></extra>"
            ),
            row=1, col=c
        )

        fig.update_xaxes(title_text="Time", row=1, col=c)
        if y_max_kw is not None:
            fig.update_yaxes(title_text="Power (kW)", range=[0, y_max_kw], row=1, col=c)
        else:
            fig.update_yaxes(title_text="Power (kW)", row=1, col=c)

    _add_power_panel(winter_day, 1)
    _add_power_panel(summer_day, 2)

    fig.update_layout(
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, x=0)
    )
    return fig




# --------- H2 FIGURE (Hourly pair) ----------
def _mk_h2_hourly_figure_pair():
    KG_PER_COULOMB = M_H2_KG_PER_MOL / (2.0 * F)
    h2_rate_kg_per_h = (
        coupling_results["Single_Stack_current"].astype(float)
        * N_cells_total * KG_PER_COULOMB * 3600.0 * FARADAIC_EFF_H2
    )
    mask_w = h2_rate_kg_per_h.index.strftime("%Y-%m-%d") == WINTER_DAY
    mask_s = h2_rate_kg_per_h.index.strftime("%Y-%m-%d") == SUMMER_DAY
    vals_for_limit = np.concatenate([
        h2_rate_kg_per_h[mask_w].values if mask_w.any() else np.array([0.0]),
        h2_rate_kg_per_h[mask_s].values if mask_s.any() else np.array([0.0])
    ])
    ymax_hourly = 1.1 * float(np.nanmax(vals_for_limit)) if vals_for_limit.size else 1.0
    ymax_hourly = max(ymax_hourly, 1e-6)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Hourly H₂ production — {WINTER_DAY}",
            f"Hourly H₂ production — {SUMMER_DAY}",
        ),
        horizontal_spacing=0.08
    )
    fig.add_trace(
        go.Bar(x=h2_rate_kg_per_h.index[mask_w], y=h2_rate_kg_per_h[mask_w].values, name="Winter day (kg/h)"),
        row=1, col=1
    )
    fig.update_yaxes(range=[0, ymax_hourly], row=1, col=1, title_text="kg/h")

    fig.add_trace(
        go.Bar(x=h2_rate_kg_per_h.index[mask_s], y=h2_rate_kg_per_h[mask_s].values, name="Summer day (kg/h)"),
        row=1, col=2
    )
    fig.update_yaxes(range=[0, ymax_hourly], row=1, col=2, title_text="kg/h")

    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=1, col=2)

    fig.update_layout(height=420, legend=dict(orientation="h", yanchor="bottom", y=-0.18, x=0))
    return fig


# --------- H2 FIGURE (Daily line only) ----------
def _mk_h2_daily_figure():
    KG_PER_COULOMB = M_H2_KG_PER_MOL / (2.0 * F)
    h2_kg_interval = (
        coupling_results["Single_Stack_current"].astype(float)
        * N_cells_total * dt_s * KG_PER_COULOMB * FARADAIC_EFF_H2
    )
    h2_kg_series = pd.Series(h2_kg_interval, index=coupling_results.index)
    h2_daily_kg = h2_kg_series.resample('D').sum()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=h2_daily_kg.index, y=h2_daily_kg.values, mode="lines+markers", name="kg/day",
                   line=dict(width=2), marker=dict(size=5))
    )
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="kg/day")
    fig.update_layout(height=420, legend=dict(orientation="h"))
    return fig

# --------- POLARIZATION CURVE FIGURE ----------
def _mk_polarization_figure_interactive():
    """
    Interactive Plotly version of the single-cell polarization curve (U–j)
    with shaded operating band and endpoint annotations.
    """
    # Safety: derive min/max j from previously computed values if available,
    # otherwise compute from coupling_results.
    try:
        j_min_seen = float(min_current_density)
    except Exception:
        _jpos = coupling_results["j_cell_Acm2"].values
        _jpos = _jpos[np.isfinite(_jpos) & (_jpos > 0)]
        j_min_seen = float(np.nanmin(_jpos)) if _jpos.size else 0.0

    try:
        j_max_seen = float(max_current_density)
    except Exception:
        _jall = coupling_results["j_cell_Acm2"].values
        _jall = _jall[np.isfinite(_jall)]
        j_max_seen = float(np.nanmax(_jall)) if _jall.size else 0.1  # fallback

    if not np.isfinite(j_max_seen) or j_max_seen <= 0:
        j_max_seen = 0.1

    j_max_plot = 1.2 * j_max_seen

    # Build the cell polarization curve U_cell(j)
    j_grid = np.linspace(0.0, j_max_plot, 600)     # A/cm²
    U_grid = u_cell_from_j(j_grid)                 # uses your existing function

    # Operating range
    j_lo = max(0.0, j_min_seen)
    j_hi = max(j_lo, j_max_seen)
    U_lo = float(u_cell_from_j(np.array([j_lo]))[0])
    U_hi = float(u_cell_from_j(np.array([j_hi]))[0])

    # Vertical offset for annotations (similar intent as your MPL version)
    U_span = float(np.nanmax(U_grid) - np.nanmin(U_grid)) if np.isfinite(np.nanmax(U_grid)) else 1.0
    dy = max(0.04, 0.08 * U_span)
    dx = 0.01 * j_max_plot

    # Plotly figure
    fig = go.Figure()

    # Main curve
    fig.add_trace(go.Scatter(
        x=j_grid, y=U_grid, mode="lines", name="Cell polarization (U–j)",
        line=dict(width=3)
    ))

    # Shaded operating range
    fig.add_vrect(
        x0=j_lo, x1=j_hi, fillcolor="orange", opacity=0.25, line_width=0,
        annotation_text="Operating range", annotation_position="top left"
    )

    # Vertical lines at endpoints
    fig.add_vline(x=j_lo, line_width=2, line_dash="dash", line_color="orange")
    fig.add_vline(x=j_hi, line_width=2, line_dash="dash", line_color="red")

    # Endpoint markers
    fig.add_trace(go.Scatter(
        x=[j_lo, j_hi], y=[U_lo, U_hi],
        mode="markers", name="Range endpoints",
        marker=dict(size=9, color="white", line=dict(width=1, color="black"))
    ))

    # Annotations slightly below the endpoints
    fig.add_annotation(x=j_lo + dx, y=U_lo - dy,
                       text=f"j_min = {j_lo:.3g} A/cm²",
                       showarrow=False, xanchor="left", yanchor="top")
    fig.add_annotation(x=j_hi - dx, y=U_hi - dy,
                       text=f"j_max = {j_hi:.3g} A/cm²",
                       showarrow=False, xanchor="right", yanchor="top")

    # Axes and layout
    fig.update_xaxes(title_text="Current density j (A/cm²)", range=[0.0, j_max_plot * 1.02],
                     showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.08)", zeroline=False)
    fig.update_yaxes(title_text="Cell voltage U_cell (V)",
                     showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.08)", zeroline=False)

    fig.update_layout(
        height=460, plot_bgcolor="white", legend=dict(orientation="h", yanchor="bottom", y=-0.1, x=0)
    )
    return fig
# -------------------------------------------------------------------------------

def _mk_cell_jU_day_pair():
    def _get_cell_day_series(day_str):
        mask = coupling_results.index.strftime("%Y-%m-%d") == day_str
        return coupling_results.loc[mask, ['j_cell_Acm2', 'U_cell_V']].copy()

    df_w = _get_cell_day_series(WINTER_DAY)
    df_s = _get_cell_day_series(SUMMER_DAY)

    def _safe_max(series):
        if series is None or series.empty:
            return 0.0
        vals = pd.to_numeric(series.replace([np.inf, -np.inf], np.nan), errors="coerce").dropna()
        return float(vals.max()) if not vals.empty else 0.0

    # Colors
    j_color = "red"
    u_color = "blue"

    # Global limits across both days (same on both subplots)
    j_max = max(_safe_max(df_w.get('j_cell_Acm2')), _safe_max(df_s.get('j_cell_Acm2')))
    u_max_data = max(_safe_max(df_w.get('U_cell_V')), _safe_max(df_s.get('U_cell_V')))
    # U-axis lower bound is U_NERNST_V; upper bound is 1.1×max(u_max_data, U_NERNST_V)
    u_lower = float(U_NERNST_V)
    u_upper = 1.1 * max(u_max_data, u_lower)
    # j-axis lower bound at 0; upper bound 1.1×max(j)
    j_upper = 1.1 * j_max if j_max > 0 else 1.0

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"secondary_y": True}, {"secondary_y": True}]],
        subplot_titles=(
            f"Electrolyzer cell — {WINTER_DAY} (Winter)",
            f"Electrolyzer cell — {SUMMER_DAY} (Summer)",
        ),
        horizontal_spacing=0.08
    )

    def _add_day(df, col, showlegend):
        if df is None or df.empty:
            fig.add_annotation(text="No coupling data", xref=f"x{col}", yref=f"y{col}",
                               x=0.5, y=0.5, showarrow=False)
            return
        # j (left axis)
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['j_cell_Acm2'].astype(float),
                mode="lines+markers", name="j (A/cm²)",
                line=dict(width=2, color=j_color),
                marker=dict(size=6, color=j_color),
                hovertemplate="Time=%{x|%H:%M}<br>j=%{y:.4f} A/cm²<extra></extra>",
                showlegend=showlegend,
            ),
            row=1, col=col, secondary_y=False
        )
        # U_cell (right axis)
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['U_cell_V'].astype(float),
                mode="lines+markers", name="U_cell (V)",
                line=dict(width=2, color=u_color),
                marker=dict(size=6, color=u_color),
                hovertemplate="Time=%{x|%H:%M}<br>U=%{y:.3f} V<extra></extra>",
                showlegend=showlegend,
            ),
            row=1, col=col, secondary_y=True
        )

    _add_day(df_w, 1, True)   # legend only once
    _add_day(df_s, 2, False)

    # X axes
    fig.update_xaxes(title_text="Time", tickformat="%H:%M", row=1, col=1)
    fig.update_xaxes(title_text="Time", tickformat="%H:%M", row=1, col=2)

    for c in (1, 2):
        # left y: current density (red)
        fig.update_yaxes(
            title_text="Cell current density j (A/cm²)",
            range=[0, j_upper],
            row=1, col=c, secondary_y=False,
            tickfont=dict(color=j_color),
            title_font=dict(color=j_color),   # <-- fix
        )
        # right y: cell voltage (blue)
        fig.update_yaxes(
            title_text="Cell voltage U_cell (V)",
            range=[u_lower, u_upper],
            row=1, col=c, secondary_y=True,
            tickfont=dict(color=u_color),
            title_font=dict(color=u_color),   # <-- fix
        )

    fig.update_layout(
        height=460,
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, x=0)
    )
    return fig



# =========================
# TOP KPIs — 6 key metrics BEFORE first figure
# =========================
top1, top2, top3 = st.columns(3)
top1.metric("PV DC energy @ MPP (kWh)", f"{E_mpp_fullR_kWh:,.1f}")
top2.metric("PV DC delivered to electrolyzer (kWh)", f"{E_coupled_kWh:,.1f}")
top3.metric("Utilization vs. MPP (%)", f"{util_pct:,.2f}")

top4, top5, top6 = st.columns(3)
top4.metric("Max electrolyzer array power (kW)", f"{Max_electrolyzer_array_power/1000.0:,.2f}")
top5.metric("Specific electricity consumption (kWh/kg H₂)", f"{SEC_kWh_per_kg:,.2f}")
top6.metric("Total H₂ produced (kg)", f"{kg_H2_total:,.3f}")

# =========================
# Render interactive figures
# =========================
st.markdown("## IV Curves — Winter & Summer Day")
st.plotly_chart(_mk_iv_figure_pair(WINTER_DAY, SUMMER_DAY), use_container_width=True)

st.markdown("## Power vs Time — Winter & Summer Day")
st.plotly_chart(_mk_power_figure_pair(WINTER_DAY, SUMMER_DAY), use_container_width=True)

st.markdown("## Hydrogen Production — Hourly Winter & Summer Day")
st.plotly_chart(_mk_h2_hourly_figure_pair(), use_container_width=True)

st.markdown("## Daily Hydrogen Production over the year")
st.plotly_chart(_mk_h2_daily_figure(), use_container_width=True)

st.markdown("## Electrolyzer-Cell Polarization Curve (U–j)")
st.plotly_chart(_mk_polarization_figure_interactive(), use_container_width=True)

# NEW: cell j & U over the day — Winter & Summer
st.markdown("## Electrolyzer Cell — j & U over the day")
st.plotly_chart(_mk_cell_jU_day_pair(), use_container_width=True)

# =========================
# DASHBOARD (bottom) — Maxima & configuration summary
# =========================
st.markdown("### PV Maxima (at coupled operation)")
pv_mod, pv_str, pv_arr = st.columns(3)
pv_mod.write(f"**Module**  \nU_max = {maxima['Single_PV_module_voltage']:.1f} V  \nI_max = {maxima['Single_PV_module_current']:.1f} A  \nP_max = {maxima['Single_PV_module_power']/1000.0:.2f} kW")
pv_str.write(f"**String**  \nU_max = {maxima['PV_string_voltage']:.1f} V  \nI_max = {maxima['PV_string_current']:.1f} A  \nP_max = {maxima['PV_string_power']/1000.0:.2f} kW")
pv_arr.write(f"**Array**  \nU_max = {maxima['PV_array_voltage']:.1f} V  \nI_max = {maxima['PV_array_current']:.1f} A  \nP_max = {maxima['PV_array_power']/1000.0:.2f} kW")

st.markdown("### Electrolyzer Maxima")
el_stack, el_string, el_array = st.columns(3)
el_stack.write(f"**Single Stack**  \nU_max = {maxima['Single_Stack_voltage']:.1f} V  \nI_max = {maxima['Single_Stack_current']:.1f} A  \nP_max = {maxima['Single_Stack_Power']/1000.0:.2f} kW")
el_string.write(f"**String**  \nU_max = {maxima['Electrolyzer_string_voltage']:.1f} V  \nI_max = {maxima['Elecgtrolyzer_string_current']:.1f} A  \nP_max = {maxima['Electrolyzer_string_power']/1000.0:.2f} kW")
el_array.write(f"**Array**  \nU_max = {maxima['Electrolyzer_array_voltage']:.1f} V  \nI_max = {maxima['Electrolyzer_array_current']:.1f} A  \nP_max = {maxima['Electrolyzer_array_power']/1000.0:.2f} kW")

# =========================
# Configuration summary (now includes details + module name/link + soiling + nameplate kWp)
# =========================
st.markdown("### Configuration Summary")
cfg1, cfg2 = st.columns(2)

with cfg1:
    st.markdown("**PV Array**")
    st.write(f"- **PV module:** [{MODULE_DISPLAY_NAME}]({MODULE_DATASHEET_URL})")
    NUmber_of_modules= N_SERIES * N_PARALLEL
    st.write(f"- Number of PV modules: **{NUmber_of_modules}**")
    st.write(f"- PV string: **{N_SERIES} modules in series**")
    st.write(f"- PV array: **{N_PARALLEL} strings in parallel**")
    # NEW: Nameplate max potential power (kWp) = 0.46 kWp per module × Ns × Np
    nameplate_kwp = 0.46 * N_SERIES * N_PARALLEL
    st.write(f"- **Max potential power: {nameplate_kwp:,.1f} kWp**  ")
    st.write(f"- Surface tilt: **{SURFACE_TILT:.1f}°**, azimuth: **{SURFACE_AZIMUTH:.1f}°**")
    st.write(f"- Inter-module jumper length (one-way): **{SEG_LEN_M:.2f} m**")
    st.write(f"- Array cable one-way length: **{ARR_CABLE_ONE_WAY_M:.1f} m**")

    # DC Cable details (moved up from Details expander)
    st.write("**DC Cable**")
    st.write(f"  • String jumper R: **{R_string:.5f} Ω** per string")
    st.write(f"  • Array cable R: **{R_array:.5f} Ω** at array level")
    st.write(f"  • Array cable CSA selected: **{ARR_CABLE_CROSS_SEC_MM2:.1f} mm²** "
             f"(required: **{ARR_CABLE_CSA_REQUIRED:.1f} mm²**)")

with cfg2:
    st.markdown("**Electrolyzer Topology**")
    st.write(f"- Cells per stack: **{N_CELLS_PER_STACK}**  |  Cell active area: **{CELL_ACTIVE_AREA_M2*1e4:.0f} cm²**")
    st.write(f"- Stacks in series per string: **{ELEC_N_SERIES_STACKS}**")
    st.write(f"- Electrolyzer strings in parallel: **{ELEC_N_PARALLEL_STRINGS}**")
    st.write(f"- U_NERNST: **{U_NERNST_V:.2f} V**, Tafel slope: **{TAFEL_SLOPE_V_DEC:.3f} V/dec**")
    st.write(f"- j₀: **{J0_A_CM2:.2e} A/cm²**, R_cell: **{R_CELL_OHM_CM2:.3f} Ω·cm²**")
    st.write(f"- Faradaic efficiency: **{FARADAIC_EFF_H2:.2f}**")

    # Electrolyzer operating range (moved up from Details expander)
    st.write("**Electrolyzer Operating Range (coupled)**")
    st.write(f"  • Minimum cell current density: **{min_current_density:,.5f} A/cm²**")
    st.write(f"  • Maximum cell current density: **{max_current_density:,.5f} A/cm²**")





