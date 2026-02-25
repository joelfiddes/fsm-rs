// FSM1 (Factorial Snow Model) — Rust port for high-performance ensemble DA.
//
// A line-by-line port of Richard Essery's FSM1 from Fortran to Rust,
// with PyO3 bindings for Python and Rayon parallelism across spatial units.
//
// Original: Richard Essery, University of Edinburgh
//   https://github.com/RichardEssery/FSM
//   Essery (2015) "A factorial snowpack model (FSM 1.0)"
//   Geosci. Model Dev. 8, 3867–3876, doi:10.5194/gmd-8-3867-2015
//
// Ported from Fortran sources: QSAT, TRIDIAG, SURF_EXCH, SURF_PROPS,
// SURF_EBAL, SNOW, SOIL, PHYSICS, CUMULATE, OUTPUT logic.

use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

// ═══════════════════════════════════════════════════════════════════════
// Physical constants (from MODULES.f90)
// ═══════════════════════════════════════════════════════════════════════

const CP: f64 = 1005.0; // Specific heat capacity of dry air (J/K/kg)
const EPS: f64 = 0.622; // Ratio of molecular weights of water and dry air
const E0: f64 = 611.213; // Saturation vapour pressure at Tm (Pa)
const G: f64 = 9.81; // Acceleration due to gravity (m/s^2)
const HCAP_ICE: f64 = 2100.0; // Specific heat capacity of ice (J/K/kg)
const HCAP_WAT: f64 = 4180.0; // Specific heat capacity of water (J/K/kg)
const HCON_AIR: f64 = 0.025; // Thermal conductivity of air (W/m/K)
const HCON_CLAY: f64 = 1.16; // Thermal conductivity of clay (W/m/K)
const HCON_ICE: f64 = 2.24; // Thermal conductivity of ice (W/m/K)
const HCON_SAND: f64 = 1.57; // Thermal conductivity of sand (W/m/K)
const HCON_WAT: f64 = 0.56; // Thermal conductivity of water (W/m/K)
const LC: f64 = 2.501e6; // Latent heat of condensation (J/kg)
const LF: f64 = 0.334e6; // Latent heat of fusion (J/kg)
const LS: f64 = LC + LF; // Latent heat of sublimation (J/kg)
const RGAS: f64 = 287.0; // Gas constant for dry air (J/K/kg)
const RWAT: f64 = 462.0; // Gas constant for water vapour (J/K/kg)
const RHO_ICE: f64 = 917.0; // Density of ice (kg/m^3)
const RHO_WAT: f64 = 1000.0; // Density of water (kg/m^3)
const SB: f64 = 5.67e-8; // Stefan-Boltzmann constant (W/m^2/K^4)
const TM: f64 = 273.15; // Melting point (K)
const VKMAN: f64 = 0.4; // Von Karman constant

// ═══════════════════════════════════════════════════════════════════════
// Grid constants
// ═══════════════════════════════════════════════════════════════════════

const NSMAX: usize = 3; // Maximum number of snow layers
const NSOIL: usize = 4; // Number of soil layers
const DZSNOW: [f64; NSMAX] = [0.1, 0.2, 0.4]; // Minimum snow layer thicknesses (m)
const DZSOIL: [f64; NSOIL] = [0.1, 0.2, 0.4, 0.8]; // Soil layer thicknesses (m)

// ═══════════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════════

#[derive(Clone)]
struct Fsm1Config {
    // Model option flags (from nconfig encoding)
    am: bool, // albedo: diagnostic(false) / prognostic(true)
    cm: bool, // conductivity: fixed(false) / density-dependent(true)
    dm: bool, // density: fixed(false) / prognostic(true)
    em: bool, // exchange: neutral(false) / stability-corrected(true)
    hm: bool, // hydraulics: free-draining(false) / bucket(true)

    // Snow parameters
    asmx: f64,  // Max fresh snow albedo (0.8)
    asmn: f64,  // Min melting snow albedo (0.5)
    hfsn: f64,  // Snow cover fraction depth scale, m (0.1)
    rhof: f64,  // Fresh snow density, kg/m³ (100)
    rcld: f64,  // Max cold snow density, kg/m³ (300)
    rmlt: f64,  // Max melting snow density, kg/m³ (500)
    trho: f64,  // Snow compaction timescale, h (200)
    salb: f64,  // Snowfall to refresh albedo, kg/m² (10)
    talb: f64,  // Albedo decay temp threshold, °C (-2)
    tcld: f64,  // Cold snow albedo decay timescale, h (1000)
    tmlt: f64,  // Melting snow albedo decay timescale, h (100)
    wirr: f64,  // Irreducible liquid water content (0.03)
    z0sn: f64,  // Snow roughness length, m (0.01)

    // Surface parameters
    alb0: f64, // Snow-free ground albedo (0.2)
    z0sf: f64, // Snow-free roughness length, m (0.1)
    bstb: f64, // Stability slope parameter (5)
    gsat: f64, // Surface conductance for saturated soil, m/s (0.01)

    // Snow thermal parameters
    kfix: f64, // Fixed snow thermal conductivity, W/m/K (0.24)
    bthr: f64, // Snow thermal conductivity exponent (2)
    rho0: f64, // Fixed snow density, kg/m³ (300)

    // Driving parameters
    dt: f64,      // Timestep (seconds)
    z_t: f64,     // Temperature measurement height (m)
    z_u: f64,     // Wind measurement height (m)
    nave: usize,  // Output averaging window
}

impl Fsm1Config {
    fn from_nconfig(nconfig: i32) -> Self {
        Fsm1Config {
            am: (nconfig / 16) % 2 == 1,
            cm: (nconfig / 8) % 2 == 1,
            dm: (nconfig / 4) % 2 == 1,
            em: (nconfig / 2) % 2 == 1,
            hm: nconfig % 2 == 1,
            // Defaults from SET_PARAMETERS.f90
            asmx: 0.8,
            asmn: 0.5,
            hfsn: 0.1,
            rhof: 100.0,
            rcld: 300.0,
            rmlt: 500.0,
            trho: 200.0,
            salb: 10.0,
            talb: -2.0,
            tcld: 1000.0,
            tmlt: 100.0,
            wirr: 0.03,
            z0sn: 0.01,
            alb0: 0.2,
            z0sf: 0.1,
            bstb: 5.0,
            gsat: 0.01,
            kfix: 0.24,
            bthr: 2.0,
            rho0: 300.0,
            dt: 3600.0,
            z_t: 2.0,
            z_u: 10.0,
            nave: 8,
        }
    }

    fn apply_overrides(&mut self, overrides: &[f64]) {
        // Order matches Python FSM1Config param_names:
        // asmx, asmn, hfsn, rhof, rcld, rmlt, trho, Salb, Talb, tcld, tmlt,
        // Wirr, z0sn, z0sf, alb0, rho0, kfix, bstb, gsat, bthr, fcly, fsnd
        let fields: [&mut f64; 20] = [
            &mut self.asmx,
            &mut self.asmn,
            &mut self.hfsn,
            &mut self.rhof,
            &mut self.rcld,
            &mut self.rmlt,
            &mut self.trho,
            &mut self.salb,
            &mut self.talb,
            &mut self.tcld,
            &mut self.tmlt,
            &mut self.wirr,
            &mut self.z0sn,
            &mut self.z0sf,
            &mut self.alb0,
            &mut self.rho0,
            &mut self.kfix,
            &mut self.bstb,
            &mut self.gsat,
            &mut self.bthr,
        ];
        for (i, val) in overrides.iter().enumerate() {
            if i < fields.len() && !val.is_nan() {
                *fields[i] = *val;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Derived soil parameters (computed once from fcly, fsnd)
// ═══════════════════════════════════════════════════════════════════════

#[derive(Clone)]
struct SoilParams {
    b: f64,         // Clapp-Hornberger exponent
    sathh: f64,     // Saturated soil water pressure (m)
    vsat: f64,      // Volumetric soil moisture at saturation
    vcrit: f64,     // Volumetric soil moisture at critical point
    hcap_soil: f64, // Volumetric heat capacity of dry soil (J/K/m^3)
    hcon_soil: f64, // Thermal conductivity of dry soil (W/m/K)
}

impl SoilParams {
    fn from_clay_sand(fcly: f64, fsnd: f64) -> Self {
        let fcly = if fcly + fsnd > 1.0 {
            1.0 - fsnd
        } else {
            fcly
        };
        let b = 3.1 + 15.7 * fcly - 0.3 * fsnd;
        let hcap_soil = (2.128 * fcly + 2.385 * fsnd) * 1e6 / (fcly + fsnd);
        let sathh = 10.0_f64.powf(0.17 - 0.63 * fcly - 1.58 * fsnd);
        let vsat = 0.505 - 0.037 * fcly - 0.142 * fsnd;
        let vcrit = vsat * (sathh / 3.364_f64).powf(1.0 / b);
        let hcon_min = HCON_CLAY.powf(fcly) * HCON_SAND.powf(1.0 - fcly);
        let hcon_soil = HCON_AIR.powf(vsat) * hcon_min.powf(1.0 - vsat);
        SoilParams {
            b,
            sathh,
            vsat,
            vcrit,
            hcap_soil,
            hcon_soil,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// State (per unit, mutable across timesteps)
// ═══════════════════════════════════════════════════════════════════════

#[derive(Clone)]
struct Fsm1State {
    tsurf: f64,             // Surface skin temperature (K)
    nsnow: usize,           // Number of snow layers
    albs: f64,              // Snow albedo
    ds: [f64; NSMAX],       // Snow layer thicknesses (m)
    sice: [f64; NSMAX],     // Ice content of snow layers (kg/m^2)
    sliq: [f64; NSMAX],     // Liquid content of snow layers (kg/m^2)
    tsnow: [f64; NSMAX],    // Snow layer temperatures (K)
    tsoil: [f64; NSOIL],    // Soil layer temperatures (K)
    theta: [f64; NSOIL],    // Volumetric soil moisture
}

impl Fsm1State {
    fn cold_start(soil: &SoilParams) -> Self {
        Fsm1State {
            tsurf: 285.0,
            nsnow: 0,
            albs: 0.8,
            ds: [0.0; NSMAX],
            sice: [0.0; NSMAX],
            sliq: [0.0; NSMAX],
            tsnow: [TM; NSMAX],
            tsoil: [285.0; NSOIL],
            theta: [0.5 * soil.vsat; NSOIL],
        }
    }

    fn pack(&self) -> Vec<f64> {
        // Pack state into a flat vector for restart capability
        // Layout: tsurf, nsnow, albs, ds[3], sice[3], sliq[3], tsnow[3], tsoil[4], theta[4]
        let mut v = Vec::with_capacity(STATE_SIZE);
        v.push(self.tsurf);
        v.push(self.nsnow as f64);
        v.push(self.albs);
        v.extend_from_slice(&self.ds);
        v.extend_from_slice(&self.sice);
        v.extend_from_slice(&self.sliq);
        v.extend_from_slice(&self.tsnow);
        v.extend_from_slice(&self.tsoil);
        v.extend_from_slice(&self.theta);
        v
    }

    fn unpack(v: &[f64]) -> Self {
        assert!(v.len() >= STATE_SIZE);
        let mut s = Fsm1State {
            tsurf: v[0],
            nsnow: v[1] as usize,
            albs: v[2],
            ds: [0.0; NSMAX],
            sice: [0.0; NSMAX],
            sliq: [0.0; NSMAX],
            tsnow: [TM; NSMAX],
            tsoil: [285.0; NSOIL],
            theta: [0.0; NSOIL],
        };
        s.ds.copy_from_slice(&v[3..3 + NSMAX]);
        s.sice.copy_from_slice(&v[3 + NSMAX..3 + 2 * NSMAX]);
        s.sliq.copy_from_slice(&v[3 + 2 * NSMAX..3 + 3 * NSMAX]);
        s.tsnow.copy_from_slice(&v[3 + 3 * NSMAX..3 + 4 * NSMAX]);
        let soil_start = 3 + 4 * NSMAX;
        s.tsoil.copy_from_slice(&v[soil_start..soil_start + NSOIL]);
        s.theta
            .copy_from_slice(&v[soil_start + NSOIL..soil_start + 2 * NSOIL]);
        s
    }
}

// tsurf(1) + nsnow(1) + albs(1) + ds(3) + sice(3) + sliq(3) + tsnow(3) + tsoil(4) + theta(4)
const STATE_SIZE: usize = 3 + 4 * NSMAX + 2 * NSOIL;

// ═══════════════════════════════════════════════════════════════════════
// Driving data (single timestep)
// ═══════════════════════════════════════════════════════════════════════

struct Driving {
    sw: f64, // Incoming shortwave radiation (W/m^2)
    lw: f64, // Incoming longwave radiation (W/m^2)
    sf: f64, // Snowfall rate (kg/m^2/s)
    rf: f64, // Rainfall rate (kg/m^2/s)
    ta: f64, // Air temperature (K)
    qa: f64, // Specific humidity (kg/kg)
    ua: f64, // Wind speed (m/s)
    ps: f64, // Surface pressure (Pa)
}

// ═══════════════════════════════════════════════════════════════════════
// Diagnostics accumulator
// ═══════════════════════════════════════════════════════════════════════

struct Diagnostics {
    sw_int: f64,         // Cumulated incoming solar radiation (J/m^2)
    sw_out: f64,         // Cumulated reflected solar radiation (J/m^2)
    sum_runoff: f64,     // Cumulated runoff (kg/m^2)
    sum_snowdepth: f64,  // Cumulated snow depth (m)
    sum_swe: f64,        // Cumulated SWE (kg/m^2)
    sum_tsurf_c: f64,    // Cumulated surface temp (°C)
    sum_tsoil2_c: f64,   // Cumulated soil layer 2 temp (°C)
}

impl Diagnostics {
    fn new() -> Self {
        Diagnostics {
            sw_int: 0.0,
            sw_out: 0.0,
            sum_runoff: 0.0,
            sum_snowdepth: 0.0,
            sum_swe: 0.0,
            sum_tsurf_c: 0.0,
            sum_tsoil2_c: 0.0,
        }
    }

    fn accumulate(
        &mut self,
        sw: f64,
        alb: f64,
        runoff: f64,
        snowdepth: f64,
        swe: f64,
        tsurf: f64,
        tsoil2: f64,
        dt: f64,
        nave: usize,
    ) {
        self.sw_int += sw * dt;
        self.sw_out += alb * sw * dt;
        self.sum_runoff += runoff * nave as f64;
        self.sum_snowdepth += snowdepth;
        self.sum_swe += swe;
        self.sum_tsurf_c += tsurf - TM;
        self.sum_tsoil2_c += tsoil2 - TM;
    }

    fn output(&self, nave: usize) -> OutputRow {
        let alb_eff = if self.sw_int > 0.0 {
            self.sw_out / self.sw_int
        } else {
            -9.0
        };
        let n = nave as f64;
        OutputRow {
            albedo: alb_eff,
            runoff: self.sum_runoff / n,
            snow_depth: self.sum_snowdepth / n,
            swe: self.sum_swe / n,
            tsurf_c: self.sum_tsurf_c / n,
            tsoil2_c: self.sum_tsoil2_c / n,
        }
    }

    fn reset(&mut self) {
        *self = Diagnostics::new();
    }
}

struct OutputRow {
    albedo: f64,
    runoff: f64,
    snow_depth: f64,
    swe: f64,
    tsurf_c: f64,
    tsoil2_c: f64,
}

// ═══════════════════════════════════════════════════════════════════════
// QSAT — Saturation specific humidity (from QSAT.f90)
// ═══════════════════════════════════════════════════════════════════════

#[inline]
fn qsat(water: bool, p: f64, t: f64) -> f64 {
    let tc = t - TM;
    let es = if tc > 0.0 || water {
        E0 * (17.5043 * tc / (241.3 + tc)).exp()
    } else {
        E0 * (22.4422 * tc / (272.186 + tc)).exp()
    };
    EPS * es / p
}

// ═══════════════════════════════════════════════════════════════════════
// TRIDIAG — Thomas algorithm (from TRIDIAG.f90)
// ═══════════════════════════════════════════════════════════════════════

fn tridiag(n: usize, a: &[f64], b: &[f64], c: &[f64], r: &[f64], x: &mut [f64]) {
    if n == 0 {
        return;
    }
    let mut gamma = vec![0.0; n];
    let mut beta = b[0];
    x[0] = r[0] / beta;
    for i in 1..n {
        gamma[i] = c[i - 1] / beta;
        beta = b[i] - a[i] * gamma[i];
        x[i] = (r[i] - a[i] * x[i - 1]) / beta;
    }
    for i in (0..n - 1).rev() {
        x[i] -= gamma[i + 1] * x[i + 1];
    }
}

// ═══════════════════════════════════════════════════════════════════════
// SURF_PROPS — Surface, surface layer and soil properties (from SURF_PROPS.f90)
// ═══════════════════════════════════════════════════════════════════════

struct SurfProps {
    alb: f64,
    csoil: [f64; NSOIL],
    dz1: f64,
    gs: f64,
    ksnow: [f64; NSMAX],
    ksoil: [f64; NSOIL],
    ksurf: f64,
    rfs: f64,
    ts1: f64,
    z0: f64,
}

fn surf_props(cfg: &Fsm1Config, soil: &SoilParams, state: &mut Fsm1State, sf: f64) -> SurfProps {
    let dt = cfg.dt;

    // Snow albedo
    if cfg.am {
        // Prognostic
        let tau = if state.tsurf >= TM {
            3600.0 * cfg.tmlt
        } else {
            3600.0 * cfg.tcld
        };
        let rt = 1.0 / tau + sf / cfg.salb;
        let alim = (cfg.asmn / tau + sf * cfg.asmx / cfg.salb) / rt;
        state.albs = alim + (state.albs - alim) * (-rt * dt).exp();
    } else {
        // Diagnostic
        state.albs = cfg.asmn + (cfg.asmx - cfg.asmn) * (state.tsurf - TM) / cfg.talb;
    }
    // Clamp albedo
    let amin = cfg.asmx.min(cfg.asmn);
    let amax = cfg.asmx.max(cfg.asmn);
    if state.albs < amin {
        state.albs = amin;
    }
    if state.albs > amax {
        state.albs = amax;
    }

    // Fresh snow density
    let rfs = if cfg.dm { cfg.rhof } else { cfg.rho0 };

    // Thermal conductivity of snow
    let mut ksnow = [cfg.kfix; NSMAX];
    if cfg.cm {
        for k in 0..state.nsnow {
            let rhos = if cfg.dm && state.ds[k] > f64::EPSILON {
                (state.sice[k] + state.sliq[k]) / state.ds[k]
            } else {
                rfs
            };
            ksnow[k] = HCON_ICE * (rhos / RHO_ICE).powf(cfg.bthr);
        }
    }

    // Partial snow cover
    let snowdepth: f64 = state.ds.iter().sum();
    let fsnow = (snowdepth / cfg.hfsn).tanh();
    let alb = fsnow * state.albs + (1.0 - fsnow) * cfg.alb0;
    let z0 = cfg.z0sn.powf(fsnow) * cfg.z0sf.powf(1.0 - fsnow);

    // Soil properties
    let dpsidt = -RHO_ICE * LF / (RHO_WAT * G * TM);
    let mut csoil = [0.0; NSOIL];
    let mut ksoil = [0.0; NSOIL];
    let mut gs = cfg.gsat; // default

    for k in 0..NSOIL {
        csoil[k] = soil.hcap_soil * DZSOIL[k];
        ksoil[k] = soil.hcon_soil;
        if state.theta[k] > f64::EPSILON {
            let mut dthudt = 0.0;
            let mut sthu = state.theta[k];
            let mut sthf = 0.0_f64;
            let tc = state.tsoil[k] - TM;
            let tmax = TM + (soil.sathh / dpsidt) * (soil.vsat / state.theta[k]).powf(soil.b);
            if state.tsoil[k] < tmax {
                dthudt = (-dpsidt * soil.vsat / (soil.b * soil.sathh))
                    * (dpsidt * tc / soil.sathh).powf(-1.0 / soil.b - 1.0);
                sthu = soil.vsat * (dpsidt * tc / soil.sathh).powf(-1.0 / soil.b);
                sthu = sthu.min(state.theta[k]);
                sthf = (state.theta[k] - sthu) * RHO_WAT / RHO_ICE;
            }
            let mf = RHO_ICE * DZSOIL[k] * sthf;
            let mu = RHO_WAT * DZSOIL[k] * sthu;
            let tc_val = tc;
            csoil[k] = soil.hcap_soil * DZSOIL[k]
                + HCAP_ICE * mf
                + HCAP_WAT * mu
                + RHO_WAT * DZSOIL[k] * ((HCAP_WAT - HCAP_ICE) * tc_val + LF) * dthudt;
            let smf = RHO_ICE * sthf / (RHO_WAT * soil.vsat);
            let smu = sthu / soil.vsat;
            let thice = if smf > 0.0 {
                soil.vsat * smf / (smu + smf)
            } else {
                0.0
            };
            let thwat = if smu > 0.0 {
                soil.vsat * smu / (smu + smf)
            } else {
                0.0
            };
            let hcon_sat =
                soil.hcon_soil * HCON_WAT.powf(thwat) * HCON_ICE.powf(thice) / HCON_AIR.powf(soil.vsat);
            ksoil[k] = (hcon_sat - soil.hcon_soil) * (smf + smu) + soil.hcon_soil;
            if k == 0 {
                gs = cfg.gsat * (smu * soil.vsat / soil.vcrit).powi(2).max(1.0);
            }
        }
    }

    // Surface layer
    let dz1 = DZSOIL[0].max(state.ds[0]);
    let mut ts1 =
        state.tsoil[0] + (state.tsnow[0] - state.tsoil[0]) * state.ds[0] / DZSOIL[0];
    let mut ksurf = DZSOIL[0]
        / (2.0 * state.ds[0] / ksnow[0] + (DZSOIL[0] - 2.0 * state.ds[0]) / ksoil[0]);
    if state.ds[0] > 0.5 * DZSOIL[0] {
        ksurf = ksnow[0];
    }
    if state.ds[0] > DZSOIL[0] {
        ts1 = state.tsnow[0];
    }

    SurfProps {
        alb,
        csoil,
        dz1,
        gs,
        ksnow,
        ksoil,
        ksurf,
        rfs,
        ts1,
        z0,
    }
}

// ═══════════════════════════════════════════════════════════════════════
// SURF_EXCH — Surface exchange coefficients (from SURF_EXCH.f90)
// ═══════════════════════════════════════════════════════════════════════

fn surf_exch(cfg: &Fsm1Config, state: &Fsm1State, z0: f64, ta: f64, ua: f64) -> f64 {
    let z_u = cfg.z_u;
    let z_ts = cfg.z_t; // zvar not used (no measurement height adjustment)

    // Neutral exchange coefficients
    let z0h = 0.1 * z0;
    let cd = (VKMAN / (z_u / z0).ln()).powi(2);
    let mut ch = VKMAN.powi(2) / ((z_u / z0).ln() * (z_ts / z0h).ln());

    // Stability correction (Louis et al. 1982)
    if cfg.em {
        let rib = G * (ta - state.tsurf) * z_u.powi(2) / (z_ts * ta * ua.powi(2));
        let fh = if rib > 0.0 {
            1.0 / (1.0 + 3.0 * cfg.bstb * rib * (1.0 + cfg.bstb * rib).sqrt())
        } else {
            1.0 - 3.0 * cfg.bstb * rib
                / (1.0 + 3.0 * cfg.bstb.powi(2) * cd * (-rib * z_u / z0).sqrt())
        };
        ch *= fh;
    }

    ch
}

// ═══════════════════════════════════════════════════════════════════════
// SURF_EBAL — Surface energy balance (from SURF_EBAL.f90)
// ═══════════════════════════════════════════════════════════════════════

struct EbalResult {
    esnow: f64,  // Snow sublimation rate (kg/m^2/s)
    gsurf: f64,  // Heat flux into surface (W/m^2)
    melt: f64,   // Surface melt rate (kg/m^2/s)
}

fn surf_ebal(
    cfg: &Fsm1Config,
    state: &mut Fsm1State,
    drv: &Driving,
    props: &SurfProps,
    ch: f64,
) -> EbalResult {
    let dt = cfg.dt;

    let mut qs = qsat(false, drv.ps, state.tsurf);
    let mut psi = props.gs / (props.gs + ch * drv.ua);
    if qs < drv.qa || state.sice[0] > 0.0 {
        psi = 1.0;
    }
    let lh = if state.tsurf > TM { LC } else { LS };
    let rho = drv.ps / (RGAS * drv.ta);
    let rkh = rho * ch * drv.ua;

    // Surface energy balance without melt
    let d_val = lh * qs / (RWAT * state.tsurf.powi(2));
    let mut esurf = psi * rkh * (qs - drv.qa);
    let mut gsurf = 2.0 * props.ksurf * (state.tsurf - props.ts1) / props.dz1;
    let mut hsurf = CP * rkh * (state.tsurf - drv.ta);
    let mut lesrf = lh * esurf;
    let mut melt = 0.0_f64;
    let mut rnet = (1.0 - props.alb) * drv.sw + drv.lw - SB * state.tsurf.powi(4);

    let mut dts = (rnet - hsurf - lesrf - gsurf)
        / ((CP + lh * psi * d_val) * rkh + 2.0 * props.ksurf / props.dz1 + 4.0 * SB * state.tsurf.powi(3));
    let mut de = psi * rkh * d_val * dts;
    let mut dg = 2.0 * props.ksurf * dts / props.dz1;
    let mut dh = CP * rkh * dts;
    let mut dr = -SB * state.tsurf.powi(3) * dts;

    // Surface melting
    if state.tsurf + dts > TM && state.sice[0] > 0.0 {
        let sice_sum: f64 = state.sice.iter().sum();
        melt = sice_sum / dt;
        dts = (rnet - hsurf - lesrf - gsurf - LF * melt)
            / ((CP + LS * psi * d_val) * rkh + 2.0 * props.ksurf / props.dz1 + 4.0 * SB * state.tsurf.powi(3));
        de = rkh * d_val * dts;
        dg = 2.0 * props.ksurf * dts / props.dz1;
        dh = CP * rkh * dts;
        if state.tsurf + dts < TM {
            qs = qsat(false, drv.ps, TM);
            esurf = rkh * (qs - drv.qa);
            gsurf = 2.0 * props.ksurf * (TM - props.ts1) / props.dz1;
            hsurf = CP * rkh * (TM - drv.ta);
            lesrf = LS * esurf;
            rnet = (1.0 - props.alb) * drv.sw + drv.lw - SB * TM.powi(4);
            melt = (rnet - hsurf - lesrf - gsurf) / LF;
            melt = melt.max(0.0);
            de = 0.0;
            dg = 0.0;
            dh = 0.0;
            dr = 0.0;
            dts = TM - state.tsurf;
        }
    }

    // Update surface temperature and fluxes
    state.tsurf += dts;
    esurf += de;
    gsurf += dg;
    hsurf += dh;
    rnet += dr;
    let _ = hsurf;
    let _ = rnet;

    let esnow = if state.sice[0] > 0.0 || state.tsurf < TM {
        lesrf = LS * esurf;
        esurf
    } else {
        lesrf = LC * esurf;
        0.0
    };
    let _ = lesrf;

    EbalResult {
        esnow,
        gsurf,
        melt,
    }
}

// ═══════════════════════════════════════════════════════════════════════
// SNOW — Snow thermodynamics and hydrology (from SNOW.f90)
// ═══════════════════════════════════════════════════════════════════════

struct SnowResult {
    gsoil: f64,
    runoff: f64,
    snowdepth: f64,
    swe: f64,
}

fn snow_step(
    cfg: &Fsm1Config,
    state: &mut Fsm1State,
    ebal: &EbalResult,
    ksnow: &[f64; NSMAX],
    ksoil: &[f64; NSOIL],
    rfs: f64,
    sf: f64,
    rf: f64,
    ta: f64,
) -> SnowResult {
    let dt = cfg.dt;
    let mut gsoil = ebal.gsurf;
    let mut roff = rf * dt;

    if state.nsnow > 0 {
        // Heat capacity
        let mut csnow = [0.0; NSMAX];
        for k in 0..state.nsnow {
            csnow[k] = state.sice[k] * HCAP_ICE + state.sliq[k] * HCAP_WAT;
        }

        // Heat conduction
        let mut dts_vec = [0.0; NSMAX];
        if state.nsnow == 1 {
            let gs0 = 2.0 / (state.ds[0] / ksnow[0] + DZSOIL[0] / ksoil[0]);
            dts_vec[0] = (ebal.gsurf + gs0 * (state.tsoil[0] - state.tsnow[0])) * dt
                / (csnow[0] + gs0 * dt);
        } else {
            let mut gs = [0.0; NSMAX];
            for k in 0..state.nsnow - 1 {
                gs[k] = 2.0 / (state.ds[k] / ksnow[k] + state.ds[k + 1] / ksnow[k + 1]);
            }
            let mut a = [0.0; NSMAX];
            let mut b = [0.0; NSMAX];
            let mut c = [0.0; NSMAX];
            let mut rhs = [0.0; NSMAX];

            a[0] = 0.0;
            b[0] = csnow[0] + gs[0] * dt;
            c[0] = -gs[0] * dt;
            rhs[0] = (ebal.gsurf - gs[0] * (state.tsnow[0] - state.tsnow[1])) * dt;

            for k in 1..state.nsnow - 1 {
                a[k] = c[k - 1];
                b[k] = csnow[k] + (gs[k - 1] + gs[k]) * dt;
                c[k] = -gs[k] * dt;
                rhs[k] = gs[k - 1] * (state.tsnow[k - 1] - state.tsnow[k]) * dt
                    + gs[k] * (state.tsnow[k + 1] - state.tsnow[k]) * dt;
            }

            let k = state.nsnow - 1;
            gs[k] = 2.0 / (state.ds[k] / ksnow[k] + DZSOIL[0] / ksoil[0]);
            a[k] = c[k - 1];
            b[k] = csnow[k] + (gs[k - 1] + gs[k]) * dt;
            c[k] = 0.0;
            rhs[k] = gs[k - 1] * (state.tsnow[k - 1] - state.tsnow[k]) * dt
                + gs[k] * (state.tsoil[0] - state.tsnow[k]) * dt;

            tridiag(state.nsnow, &a, &b, &c, &rhs, &mut dts_vec);
        }

        for k in 0..state.nsnow {
            state.tsnow[k] += dts_vec[k];
        }

        // Gsoil from bottom snow layer
        let kb = state.nsnow - 1;
        let gs_bot = 2.0 / (state.ds[kb] / ksnow[kb] + DZSOIL[0] / ksoil[0]);
        gsoil = gs_bot * (state.tsnow[kb] - state.tsoil[0]);

        // Convert melting ice to liquid water
        let mut dsice = ebal.melt * dt;
        for k in 0..state.nsnow {
            let coldcont = csnow[k] * (TM - state.tsnow[k]);
            if coldcont < 0.0 {
                dsice -= coldcont / LF;
                state.tsnow[k] = TM;
            }
            if dsice > 0.0 {
                if dsice > state.sice[k] {
                    // Layer melts completely
                    dsice -= state.sice[k];
                    state.ds[k] = 0.0;
                    state.sliq[k] += state.sice[k];
                    state.sice[k] = 0.0;
                } else {
                    // Layer melts partially
                    state.ds[k] = (1.0 - dsice / state.sice[k]) * state.ds[k];
                    state.sice[k] -= dsice;
                    state.sliq[k] += dsice;
                    dsice = 0.0;
                }
            }
        }

        // Remove snow by sublimation
        let mut dsice_subl = ebal.esnow.max(0.0) * dt;
        if dsice_subl > 0.0 {
            for k in 0..state.nsnow {
                if dsice_subl > state.sice[k] {
                    dsice_subl -= state.sice[k];
                    state.ds[k] = 0.0;
                    state.sice[k] = 0.0;
                } else {
                    state.ds[k] = (1.0 - dsice_subl / state.sice[k]) * state.ds[k];
                    state.sice[k] -= dsice_subl;
                    dsice_subl = 0.0;
                }
            }
        }

        // Snow hydraulics
        if cfg.hm {
            // Bucket storage
            // Recompute csnow after melt/sublimation
            for k in 0..state.nsnow {
                csnow[k] = state.sice[k] * HCAP_ICE + state.sliq[k] * HCAP_WAT;
            }
            for k in 0..state.nsnow {
                let phi = if state.ds[k] > f64::EPSILON {
                    1.0 - state.sice[k] / (RHO_ICE * state.ds[k])
                } else {
                    0.0
                };
                let sliq_max = RHO_WAT * state.ds[k] * phi * cfg.wirr;
                state.sliq[k] += roff;
                roff = 0.0;
                if state.sliq[k] > sliq_max {
                    roff = state.sliq[k] - sliq_max;
                    state.sliq[k] = sliq_max;
                }
                let coldcont = csnow[k] * (TM - state.tsnow[k]);
                if coldcont > 0.0 {
                    let dsice_freeze = state.sliq[k].min(coldcont / LF);
                    state.sliq[k] -= dsice_freeze;
                    state.sice[k] += dsice_freeze;
                    state.tsnow[k] += LF * dsice_freeze / csnow[k];
                }
            }
        } else {
            // Free-draining snow
            for k in 0..state.nsnow {
                roff += state.sliq[k];
                state.sliq[k] = 0.0;
            }
        }

        // Snow compaction
        if cfg.dm {
            // Prognostic density
            let tau = 3600.0 * cfg.trho;
            for k in 0..state.nsnow {
                if state.ds[k] > f64::EPSILON {
                    let mut rhos = (state.sice[k] + state.sliq[k]) / state.ds[k];
                    if state.tsnow[k] >= TM {
                        if rhos < cfg.rmlt {
                            rhos = cfg.rmlt + (rhos - cfg.rmlt) * (-dt / tau).exp();
                        }
                    } else if rhos < cfg.rcld {
                        rhos = cfg.rcld + (rhos - cfg.rcld) * (-dt / tau).exp();
                    }
                    state.ds[k] = (state.sice[k] + state.sliq[k]) / rhos;
                }
            }
        } else {
            // Fixed density
            for k in 0..state.nsnow {
                state.ds[k] = (state.sice[k] + state.sliq[k]) / cfg.rho0;
            }
        }
    } // end existing snowpack

    // Add snowfall and frost to layer 1
    let dsice_new = sf * dt - ebal.esnow.min(0.0) * dt;
    state.ds[0] += dsice_new / rfs;
    state.sice[0] += dsice_new;

    // New snowpack
    if state.nsnow == 0 && state.sice[0] > 0.0 {
        state.nsnow = 1;
        state.tsnow[0] = ta.min(TM);
    }

    // Calculate snow depth and SWE
    let mut snowdepth = 0.0;
    let mut swe = 0.0;
    for k in 0..state.nsnow {
        snowdepth += state.ds[k];
        swe += state.sice[k] + state.sliq[k];
    }

    // Store state of old layers
    let mut d_old = [0.0; NSMAX];
    let mut s_old = [0.0; NSMAX];
    let mut w_old = [0.0; NSMAX];
    let mut e_old = [0.0; NSMAX];
    for k in 0..NSMAX {
        d_old[k] = state.ds[k];
        s_old[k] = state.sice[k];
        w_old[k] = state.sliq[k];
    }
    for k in 0..state.nsnow {
        let csnow_k = state.sice[k] * HCAP_ICE + state.sliq[k] * HCAP_WAT;
        e_old[k] = csnow_k * (state.tsnow[k] - TM);
    }
    let nold = state.nsnow;

    // Initialize new layers
    for k in 0..NSMAX {
        state.ds[k] = 0.0;
        state.sice[k] = 0.0;
        state.sliq[k] = 0.0;
        state.tsnow[k] = TM;
    }
    let mut u_new = [0.0; NSMAX];
    state.nsnow = 0;

    if swe > 0.0 {
        // Re-assign and count snow layers
        let mut dnew = snowdepth;
        state.ds[0] = dnew;
        let mut k = 0;
        if state.ds[0] > DZSNOW[0] {
            for ki in 0..NSMAX {
                k = ki;
                state.ds[ki] = DZSNOW[ki];
                dnew -= DZSNOW[ki];
                if dnew <= DZSNOW[ki] || ki == NSMAX - 1 {
                    state.ds[ki] += dnew;
                    break;
                }
            }
        }
        state.nsnow = k + 1;

        // Fill new layers from top downwards
        let mut knew = 0;
        let mut dnew = state.ds[0];
        for kold in 0..nold {
            loop {
                if d_old[kold] < dnew {
                    // Transfer all snow from old layer
                    state.sice[knew] += s_old[kold];
                    state.sliq[knew] += w_old[kold];
                    u_new[knew] += e_old[kold];
                    dnew -= d_old[kold];
                    break;
                } else {
                    // Transfer some snow from old layer
                    let wt = dnew / d_old[kold];
                    state.sice[knew] += wt * s_old[kold];
                    state.sliq[knew] += wt * w_old[kold];
                    u_new[knew] += wt * e_old[kold];
                    d_old[kold] = (1.0 - wt) * d_old[kold];
                    e_old[kold] = (1.0 - wt) * e_old[kold];
                    s_old[kold] = (1.0 - wt) * s_old[kold];
                    w_old[kold] = (1.0 - wt) * w_old[kold];
                    knew += 1;
                    if knew >= state.nsnow {
                        break;
                    }
                    dnew = state.ds[knew];
                }
            }
            if knew >= state.nsnow {
                break;
            }
        }

        // Diagnose snow layer temperatures
        for k in 0..state.nsnow {
            let csnow_k = state.sice[k] * HCAP_ICE + state.sliq[k] * HCAP_WAT;
            if csnow_k > f64::EPSILON {
                state.tsnow[k] = TM + u_new[k] / csnow_k;
            }
        }
    }

    SnowResult {
        gsoil,
        runoff: roff,
        snowdepth,
        swe,
    }
}

// ═══════════════════════════════════════════════════════════════════════
// SOIL — Update soil temperatures (from SOIL.f90)
// ═══════════════════════════════════════════════════════════════════════

fn soil_step(
    dt: f64,
    state: &mut Fsm1State,
    csoil: &[f64; NSOIL],
    gsoil: f64,
    ksoil: &[f64; NSOIL],
) {
    let mut gs = [0.0; NSOIL];
    for k in 0..NSOIL - 1 {
        gs[k] = 2.0 / (DZSOIL[k] / ksoil[k] + DZSOIL[k + 1] / ksoil[k + 1]);
    }

    let mut a = [0.0; NSOIL];
    let mut b = [0.0; NSOIL];
    let mut c = [0.0; NSOIL];
    let mut rhs = [0.0; NSOIL];

    a[0] = 0.0;
    b[0] = csoil[0] + gs[0] * dt;
    c[0] = -gs[0] * dt;
    rhs[0] = (gsoil - gs[0] * (state.tsoil[0] - state.tsoil[1])) * dt;

    for k in 1..NSOIL - 1 {
        a[k] = c[k - 1];
        b[k] = csoil[k] + (gs[k - 1] + gs[k]) * dt;
        c[k] = -gs[k] * dt;
        rhs[k] = gs[k - 1] * (state.tsoil[k - 1] - state.tsoil[k]) * dt
            + gs[k] * (state.tsoil[k + 1] - state.tsoil[k]) * dt;
    }

    let k = NSOIL - 1;
    gs[k] = ksoil[k] / DZSOIL[k];
    a[k] = c[k - 1];
    b[k] = csoil[k] + (gs[k - 1] + gs[k]) * dt;
    c[k] = 0.0;
    rhs[k] = gs[k - 1] * (state.tsoil[k - 1] - state.tsoil[k]) * dt;

    let mut dts = [0.0; NSOIL];
    tridiag(NSOIL, &a, &b, &c, &rhs, &mut dts);

    for k in 0..NSOIL {
        state.tsoil[k] += dts[k];
    }
}

// ═══════════════════════════════════════════════════════════════════════
// PHYSICS — Single timestep driver (from PHYSICS.f90)
// ═══════════════════════════════════════════════════════════════════════

struct PhysicsResult {
    alb: f64,
    runoff: f64,
    snowdepth: f64,
    swe: f64,
}

fn physics_step(
    cfg: &Fsm1Config,
    soil: &SoilParams,
    state: &mut Fsm1State,
    drv: &Driving,
) -> PhysicsResult {
    let props = surf_props(cfg, soil, state, drv.sf);

    // 6-iteration convergence loop (SURF_EXCH + SURF_EBAL)
    let mut ebal = EbalResult {
        esnow: 0.0,
        gsurf: 0.0,
        melt: 0.0,
    };
    for _ in 0..6 {
        let ch = surf_exch(cfg, state, props.z0, drv.ta, drv.ua);
        ebal = surf_ebal(cfg, state, drv, &props, ch);
    }

    let snow = snow_step(cfg, state, &ebal, &props.ksnow, &props.ksoil, props.rfs, drv.sf, drv.rf, drv.ta);

    soil_step(cfg.dt, state, &props.csoil, snow.gsoil, &props.ksoil);

    PhysicsResult {
        alb: props.alb,
        runoff: snow.runoff,
        snowdepth: snow.snowdepth,
        swe: snow.swe,
    }
}

// ═══════════════════════════════════════════════════════════════════════
// run_unit — Full simulation for a single unit
// ═══════════════════════════════════════════════════════════════════════

fn run_unit(
    cfg: &Fsm1Config,
    soil: &SoilParams,
    initial_state: Option<&[f64]>,
    forcing: &[Driving],
    nave: usize,
) -> (Vec<OutputRow>, Vec<f64>) {
    let mut state = match initial_state {
        Some(s) => Fsm1State::unpack(s),
        None => Fsm1State::cold_start(soil),
    };

    let n_time = forcing.len();
    let n_out = n_time / nave;
    let mut outputs = Vec::with_capacity(n_out);
    let mut diags = Diagnostics::new();

    let mut step_in_window = 0;
    for t in 0..n_time {
        let drv = &forcing[t];
        let result = physics_step(cfg, soil, &mut state, drv);

        // Get tsoil[1] for output (layer index 1, the second layer)
        let tsoil2 = if NSOIL >= 2 {
            state.tsoil[1]
        } else {
            state.tsoil[0]
        };

        diags.accumulate(
            drv.sw,
            result.alb,
            result.runoff,
            result.snowdepth,
            result.swe,
            state.tsurf,
            tsoil2,
            cfg.dt,
            nave,
        );

        step_in_window += 1;
        if step_in_window == nave {
            outputs.push(diags.output(nave));
            diags.reset();
            step_in_window = 0;
        }
    }

    let final_state = state.pack();
    (outputs, final_state)
}

// ═══════════════════════════════════════════════════════════════════════
// PyO3 entry point — batch all units in parallel
// ═══════════════════════════════════════════════════════════════════════

/// Run FSM1 for all units in parallel.
///
/// Parameters
/// ----------
/// sw : ndarray (n_time, n_units)
///     Incoming shortwave radiation (W/m²)
/// lw : ndarray (n_time, n_units)
///     Incoming longwave radiation (W/m²)
/// sf : ndarray (n_time, n_units)
///     Snowfall rate (kg/m²/s)
/// rf : ndarray (n_time, n_units)
///     Rainfall rate (kg/m²/s)
/// ta : ndarray (n_time, n_units)
///     Air temperature (K)
/// rh : ndarray (n_time, n_units)
///     Relative humidity (%)
/// ua : ndarray (n_time, n_units)
///     Wind speed (m/s)
/// ps : ndarray (n_time, n_units)
///     Surface pressure (Pa)
/// nconfig : i32
///     5-bit model option encoding (default 31 = all options on)
/// dt : f64
///     Timestep in seconds
/// nave : usize
///     Output averaging window
/// z_t : f64
///     Temperature measurement height (m)
/// z_u : f64
///     Wind measurement height (m)
/// param_overrides : ndarray (22,)
///     Parameter overrides: [asmx, asmn, hfsn, rhof, rcld, rmlt, trho,
///     Salb, Talb, tcld, tmlt, Wirr, z0sn, z0sf, alb0, rho0, kfix,
///     bstb, gsat, bthr, fcly, fsnd]. NaN = use default.
/// initial_state : ndarray (n_units, STATE_SIZE), optional
///     Initial state per unit. None = cold start.
///
/// Returns
/// -------
/// tuple of 7 ndarrays:
///     swe (n_out, n_units), snow_depth, surface_temp_C, soil_temp_C,
///     runoff, albedo, final_state (n_units, STATE_SIZE)
#[pyfunction]
#[pyo3(signature = (sw, lw, sf, rf, ta, rh, ua, ps, nconfig, dt, nave, z_t, z_u, param_overrides, initial_state=None))]
fn run_fsm1_batch<'py>(
    py: Python<'py>,
    sw: PyReadonlyArray2<'py, f64>,
    lw: PyReadonlyArray2<'py, f64>,
    sf: PyReadonlyArray2<'py, f64>,
    rf: PyReadonlyArray2<'py, f64>,
    ta: PyReadonlyArray2<'py, f64>,
    rh: PyReadonlyArray2<'py, f64>,
    ua: PyReadonlyArray2<'py, f64>,
    ps: PyReadonlyArray2<'py, f64>,
    nconfig: i32,
    dt: f64,
    nave: usize,
    z_t: f64,
    z_u: f64,
    param_overrides: PyReadonlyArray1<'py, f64>,
    initial_state: Option<PyReadonlyArray2<'py, f64>>,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
)> {
    let sw = sw.as_array();
    let lw = lw.as_array();
    let sf = sf.as_array();
    let rf = rf.as_array();
    let ta = ta.as_array();
    let rh = rh.as_array();
    let ua = ua.as_array();
    let ps = ps.as_array();
    let overrides = param_overrides.as_array();

    let n_time = sw.shape()[0];
    let n_units = sw.shape()[1];
    let n_out = n_time / nave;

    // Build config
    let mut cfg = Fsm1Config::from_nconfig(nconfig);
    cfg.dt = dt;
    cfg.nave = nave;
    cfg.z_t = z_t;
    cfg.z_u = z_u;

    // Apply parameter overrides (first 20 are model params)
    let overrides_slice = overrides.as_slice().unwrap();
    cfg.apply_overrides(overrides_slice);

    // Extract soil params (last 2 of overrides: fcly, fsnd)
    let fcly = if overrides_slice.len() > 20 && !overrides_slice[20].is_nan() {
        overrides_slice[20]
    } else {
        0.3
    };
    let fsnd = if overrides_slice.len() > 21 && !overrides_slice[21].is_nan() {
        overrides_slice[21]
    } else {
        0.6
    };
    let soil = SoilParams::from_clay_sand(fcly, fsnd);

    // Pre-extract initial state if provided
    let init_state_arr = initial_state.as_ref().map(|is| is.as_array().to_owned());

    // Pre-extract forcing into column-major per-unit slices for cache locality
    // Each unit gets its own Vec<Driving>
    let unit_forcings: Vec<Vec<Driving>> = (0..n_units)
        .map(|u| {
            (0..n_time)
                .map(|t| {
                    // Convert RH% to specific humidity: Qa = (RH/100) * Qsat(water=true, Ps, Ta)
                    let ta_val = ta[[t, u]];
                    let ps_val = ps[[t, u]];
                    let rh_val = rh[[t, u]];
                    let qs = qsat(true, ps_val, ta_val);
                    let qa = (rh_val / 100.0) * qs;
                    Driving {
                        sw: sw[[t, u]],
                        lw: lw[[t, u]],
                        sf: sf[[t, u]],
                        rf: rf[[t, u]],
                        ta: ta_val,
                        qa,
                        ua: ua[[t, u]].max(0.1),
                        ps: ps_val,
                    }
                })
                .collect()
        })
        .collect();

    // Run all units in parallel via rayon
    let results: Vec<(Vec<OutputRow>, Vec<f64>)> = py.allow_threads(|| {
        unit_forcings
            .par_iter()
            .enumerate()
            .map(|(u, forcing)| {
                let init = init_state_arr.as_ref().map(|arr| {
                    let row: Vec<f64> = (0..STATE_SIZE).map(|i| arr[[u, i]]).collect();
                    row
                });
                let init_ref = init.as_deref();
                run_unit(&cfg, &soil, init_ref, forcing, nave)
            })
            .collect()
    });

    // Collect results into output arrays
    let mut swe_out = ndarray::Array2::<f64>::zeros((n_out, n_units));
    let mut depth_out = ndarray::Array2::<f64>::zeros((n_out, n_units));
    let mut tsurf_out = ndarray::Array2::<f64>::zeros((n_out, n_units));
    let mut tsoil_out = ndarray::Array2::<f64>::zeros((n_out, n_units));
    let mut runoff_out = ndarray::Array2::<f64>::zeros((n_out, n_units));
    let mut albedo_out = ndarray::Array2::<f64>::zeros((n_out, n_units));
    let mut state_out = ndarray::Array2::<f64>::zeros((n_units, STATE_SIZE));

    for (u, (rows, final_state)) in results.iter().enumerate() {
        for (t, row) in rows.iter().enumerate() {
            if t < n_out {
                swe_out[[t, u]] = row.swe;
                depth_out[[t, u]] = row.snow_depth;
                tsurf_out[[t, u]] = row.tsurf_c;
                tsoil_out[[t, u]] = row.tsoil2_c;
                runoff_out[[t, u]] = row.runoff;
                albedo_out[[t, u]] = row.albedo;
            }
        }
        for (i, &val) in final_state.iter().enumerate() {
            if i < STATE_SIZE {
                state_out[[u, i]] = val;
            }
        }
    }

    Ok((
        PyArray2::from_owned_array_bound(py, swe_out),
        PyArray2::from_owned_array_bound(py, depth_out),
        PyArray2::from_owned_array_bound(py, tsurf_out),
        PyArray2::from_owned_array_bound(py, tsoil_out),
        PyArray2::from_owned_array_bound(py, runoff_out),
        PyArray2::from_owned_array_bound(py, albedo_out),
        PyArray2::from_owned_array_bound(py, state_out),
    ))
}

/// Get the state vector size for serialization.
#[pyfunction]
fn state_size() -> usize {
    STATE_SIZE
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_fsm1_batch, m)?)?;
    m.add_function(wrap_pyfunction!(state_size, m)?)?;
    Ok(())
}
