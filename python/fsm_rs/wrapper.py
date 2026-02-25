"""High-level numpy wrapper around the Rust FSM1 batch runner."""

from __future__ import annotations

import numpy as np

from fsm_rs._fsm_rs import run_fsm1_batch


def run_fsm1(
    sw: np.ndarray,
    lw: np.ndarray,
    sf: np.ndarray,
    rf: np.ndarray,
    ta: np.ndarray,
    rh: np.ndarray,
    ua: np.ndarray,
    ps: np.ndarray,
    *,
    nconfig: int = 31,
    dt: float = 3600.0,
    nave: int = 24,
    z_t: float = 2.0,
    z_u: float = 10.0,
    params: dict | None = None,
    initial_state: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Run FSM1 for one or more spatial units.

    Parameters
    ----------
    sw, lw, sf, rf, ta, rh, ua, ps : ndarray
        Forcing arrays. Shape (n_time,) for a single unit or (n_time, n_units)
        for a batch.  See units below.
    nconfig : int
        5-bit model option encoding (0–31). Default 31 = all options on.
        Bits (MSB → LSB): Albedo, Conductivity, Density, Exchange, Hydraulics.
    dt : float
        Timestep in seconds (default 3600).
    nave : int
        Output averaging window in timesteps (default 24 → daily for hourly).
    z_t : float
        Temperature/humidity measurement height (m). Default 2.
    z_u : float
        Wind measurement height (m). Default 10.
    params : dict, optional
        Override default parameters. Keys: asmx, asmn, hfsn, rhof, rcld,
        rmlt, trho, Salb, Talb, tcld, tmlt, Wirr, z0sn, z0sf, alb0, rho0,
        kfix, bstb, gsat, bthr, fcly, fsnd.
    initial_state : ndarray (n_units, 23), optional
        Restart state from a previous run's ``final_state``.

    Returns
    -------
    dict with keys:
        swe          : (n_out, n_units) snow water equivalent (kg/m²)
        snow_depth   : (n_out, n_units) snow depth (m)
        surface_temp : (n_out, n_units) surface temperature (°C)
        soil_temp    : (n_out, n_units) soil layer-2 temperature (°C)
        runoff       : (n_out, n_units) runoff (kg/m²)
        albedo       : (n_out, n_units) effective albedo
        final_state  : (n_units, 23) restart state vector

    Units
    -----
    sw : W/m²   (incoming shortwave)
    lw : W/m²   (incoming longwave)
    sf : kg/m²/s (snowfall rate)
    rf : kg/m²/s (rainfall rate)
    ta : K       (air temperature)
    rh : %       (relative humidity, 0–100)
    ua : m/s     (wind speed)
    ps : Pa      (surface pressure)
    """
    # Promote 1-D (single unit) to 2-D
    single = sw.ndim == 1
    if single:
        sw = sw[:, None]
        lw = lw[:, None]
        sf = sf[:, None]
        rf = rf[:, None]
        ta = ta[:, None]
        rh = rh[:, None]
        ua = ua[:, None]
        ps = ps[:, None]

    # Ensure contiguous float64
    sw = np.ascontiguousarray(sw, dtype=np.float64)
    lw = np.ascontiguousarray(lw, dtype=np.float64)
    sf = np.ascontiguousarray(sf, dtype=np.float64)
    rf = np.ascontiguousarray(rf, dtype=np.float64)
    ta = np.ascontiguousarray(ta, dtype=np.float64)
    rh = np.ascontiguousarray(rh, dtype=np.float64)
    ua = np.ascontiguousarray(ua, dtype=np.float64)
    ps = np.ascontiguousarray(ps, dtype=np.float64)

    # Build parameter override vector (22 elements, NaN = use default)
    param_names = [
        "asmx", "asmn", "hfsn", "rhof", "rcld", "rmlt", "trho",
        "Salb", "Talb", "tcld", "tmlt", "Wirr", "z0sn", "z0sf",
        "alb0", "rho0", "kfix", "bstb", "gsat", "bthr", "fcly", "fsnd",
    ]
    overrides = np.full(22, np.nan)
    if params:
        for i, name in enumerate(param_names):
            if name in params:
                overrides[i] = params[name]

    swe, depth, tsurf, tsoil, runoff, albedo, final_state = run_fsm1_batch(
        sw, lw, sf, rf, ta, rh, ua, ps,
        nconfig, dt, nave, z_t, z_u,
        overrides, initial_state,
    )

    result = {
        "swe": swe,
        "snow_depth": depth,
        "surface_temp": tsurf,
        "soil_temp": tsoil,
        "runoff": runoff,
        "albedo": albedo,
        "final_state": final_state,
    }

    # Squeeze back to 1-D if single unit
    if single:
        for k in ("swe", "snow_depth", "surface_temp", "soil_temp", "runoff", "albedo"):
            result[k] = result[k][:, 0]
        result["final_state"] = result["final_state"][0, :]

    return result
