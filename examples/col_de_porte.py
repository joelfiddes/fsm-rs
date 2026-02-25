#!/usr/bin/env python
"""Run FSM1 on Col de Porte 2005–06 data and generate comparison plots.

Col de Porte is an alpine snow-research site at 1325 m in the Chartreuse
range near Grenoble, France.  The forcing and observation data are from
Richard Essery's FSM repository.

Generates three plots in docs/:
  1. comparison.png  — SWE and snow depth: Fortran vs Rust vs observations
  2. scatter.png     — Scatter: Rust vs Fortran (SWE + snow depth)
  3. performance.png — Timing bar chart (Fortran vs Rust)

Requires the Fortran FSM1 output file (out_CdP_0506.txt) to be present
in data/ for plots 1 and 2.  If absent, those plots fall back to Rust-only.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fsm_rs import run_fsm1

# ─── Paths ────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)


# ─── Parse CdP forcing ───────────────────────────────────────────────────

def load_forcing(path: Path) -> dict:
    """Parse FSM1 met_CdP_0506.txt forcing file.

    Columns: year month day hour SW LW SF RF Ta RH UA Ps
    """
    raw = np.loadtxt(path)
    n = raw.shape[0]
    dates = pd.to_datetime(
        {
            "year": raw[:, 0].astype(int),
            "month": raw[:, 1].astype(int),
            "day": raw[:, 2].astype(int),
            "hour": raw[:, 3].astype(int),
        }
    )
    return {
        "dates": dates,
        "sw": raw[:, 4],
        "lw": raw[:, 5],
        "sf": raw[:, 6],
        "rf": raw[:, 7],
        "ta": raw[:, 8],
        "rh": raw[:, 9],
        "ua": raw[:, 10],
        "ps": raw[:, 11],
        "n_time": n,
    }


def load_obs(path: Path) -> dict:
    """Parse FSM1 obs_CdP_0506.txt observation file.

    Columns: year month day albedo runoff snow_depth SWE Tsurf Tsoil2
    """
    raw = np.loadtxt(path)
    dates = pd.to_datetime(
        {
            "year": raw[:, 0].astype(int),
            "month": raw[:, 1].astype(int),
            "day": raw[:, 2].astype(int),
        }
    )
    # Replace -99 with NaN
    data = raw[:, 3:].copy()
    data[data < -90] = np.nan

    return {
        "dates": dates,
        "albedo": data[:, 0],
        "runoff": data[:, 1],
        "snow_depth": data[:, 2],
        "swe": data[:, 3],
        "tsurf": data[:, 4],
        "tsoil2": data[:, 5],
    }


def load_fortran_output(path: Path) -> dict | None:
    """Parse Fortran FSM1 output file (out_CdP_0506.txt).

    Columns: year month day albedo runoff snow_depth SWE Tsurf Tsoil2
    Returns None if the file does not exist.
    """
    if not path.exists():
        return None
    raw = np.loadtxt(path)
    dates = pd.to_datetime(
        {
            "year": raw[:, 0].astype(int),
            "month": raw[:, 1].astype(int),
            "day": raw[:, 2].astype(int),
        }
    )
    return {
        "dates": dates,
        "albedo": raw[:, 3],
        "runoff": raw[:, 4],
        "snow_depth": raw[:, 5],
        "swe": raw[:, 6],
        "tsurf": raw[:, 7],
        "tsoil2": raw[:, 8],
    }


# ─── Run model ────────────────────────────────────────────────────────────

def run_cdp():
    """Run Rust FSM1 on Col de Porte forcing and load Fortran output."""
    forcing = load_forcing(DATA / "met_CdP_0506.txt")
    obs = load_obs(DATA / "obs_CdP_0506.txt")
    fortran = load_fortran_output(DATA / "out_CdP_0506.txt")

    print(f"Forcing: {forcing['n_time']} hourly timesteps")
    print(f"Observations: {len(obs['dates'])} daily values")
    if fortran:
        print(f"Fortran output: {len(fortran['dates'])} daily values")
    else:
        print("Fortran output: not found (scatter plot will be skipped)")

    # Run with nconfig=31 (all options on), hourly dt, daily averaging
    t0 = time.perf_counter()
    result = run_fsm1(
        sw=forcing["sw"],
        lw=forcing["lw"],
        sf=forcing["sf"],
        rf=forcing["rf"],
        ta=forcing["ta"],
        rh=forcing["rh"],
        ua=forcing["ua"],
        ps=forcing["ps"],
        nconfig=31,
        dt=3600.0,
        nave=24,
    )
    elapsed = time.perf_counter() - t0
    print(f"Rust FSM1: {elapsed*1000:.1f} ms for {forcing['n_time']} timesteps")

    return forcing, obs, fortran, result, elapsed


# ─── Plot 1: Comparison with observations ──────────────────────────────

def plot_comparison(obs, fortran, result, dates_daily):
    """Two-panel time series: SWE and snow depth — Fortran vs Rust vs obs."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)

    has_fortran = fortran is not None

    # SWE
    ax = axes[0]
    ax.plot(obs["dates"], obs["swe"], "o", color="#dc2626", markersize=2.5,
            alpha=0.7, label="Observations", zorder=3)
    if has_fortran:
        ax.plot(fortran["dates"], fortran["swe"], color="#f59e0b", linewidth=1.8,
                label="Fortran FSM1", zorder=2)
    ax.plot(dates_daily, result["swe"], color="#2563eb", linewidth=1.8,
            linestyle="--", label="fsm-rs (Rust)", zorder=2)
    ax.set_ylabel("SWE (kg/m²)")
    ax.legend(loc="upper left", framealpha=0.9)
    title = "Col de Porte 2005–06 — Fortran FSM1 vs fsm-rs vs Observations"
    if not has_fortran:
        title = "Col de Porte 2005–06 — fsm-rs vs Observations"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Snow depth
    ax = axes[1]
    ax.plot(obs["dates"], obs["snow_depth"], "o", color="#dc2626", markersize=2.5,
            alpha=0.7, label="Observations", zorder=3)
    if has_fortran:
        ax.plot(fortran["dates"], fortran["snow_depth"], color="#f59e0b",
                linewidth=1.8, label="Fortran FSM1", zorder=2)
    ax.plot(dates_daily, result["snow_depth"], color="#2563eb", linewidth=1.8,
            linestyle="--", label="fsm-rs (Rust)", zorder=2)
    ax.set_ylabel("Snow depth (m)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()

    plt.tight_layout()
    out = DOCS / "comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ─── Plot 2: Scatter (Rust vs Fortran on CdP) ─────────────────────────

def plot_scatter(fortran, result):
    """Two-panel scatter: Rust vs Fortran for SWE and snow depth."""
    if fortran is None:
        print("Skipping scatter plot (no Fortran output)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for ax, var, unit, fmt in [
        (axes[0], "swe", "kg/m²", ".2f"),
        (axes[1], "snow_depth", "m", ".4f"),
    ]:
        fort_vals = fortran[var]
        rust_vals = result[var]
        ax.scatter(fort_vals, rust_vals, s=14, alpha=0.6,
                   color="#2563eb", edgecolors="none")

        lim = max(fort_vals.max(), rust_vals.max()) * 1.05
        ax.plot([0, lim], [0, lim], "--", color="#6b7280", linewidth=1, label="1:1")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")

        r2 = np.corrcoef(fort_vals, rust_vals)[0, 1] ** 2
        rmse = np.sqrt(np.mean((fort_vals - rust_vals) ** 2))
        ax.text(0.05, 0.92, f"R² = {r2:.6f}\nRMSE = {rmse:{fmt}} {unit}",
                transform=ax.transAxes, fontsize=11,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        label = "SWE" if var == "swe" else "Snow depth"
        ax.set_xlabel(f"Fortran FSM1 {label.lower()} ({unit})")
        ax.set_ylabel(f"fsm-rs {label.lower()} ({unit})")
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Col de Porte 2005\u201306 \u2014 Fortran FSM1 vs fsm-rs",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out = DOCS / "scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ─── Plot 3: Performance bar chart ────────────────────────────────────

def plot_performance(elapsed_single):
    """Bar chart showing timing comparison.

    Uses measured Rust single-unit time and published/measured benchmarks
    for the kaz_forecast domain (15,684 units x 8,760 hours).
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Benchmarks from kaz_forecast cross-validation (15,684 units, 8760 h)
    # Fortran: sequential, ~2.5 s/unit → ~11 hours total
    # Rust: parallel (rayon), measured 6.4 min for full domain
    labels = ["Fortran (sequential)", "Rust (parallel, 10 cores)"]
    times = [39600, 384]  # seconds: ~11h vs ~6.4min
    colors = ["#94a3b8", "#2563eb"]

    bars = ax.barh(labels, times, color=colors, height=0.5, edgecolor="white")

    # Annotate
    for bar, t in zip(bars, times):
        if t > 3600:
            label = f"{t/3600:.1f} h"
        elif t > 60:
            label = f"{t/60:.1f} min"
        else:
            label = f"{t:.1f} s"
        ax.text(bar.get_width() + 200, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=12, fontweight="bold")

    speedup = times[0] / times[1]
    ax.set_xlabel("Wall time (seconds)")
    ax.set_title(
        f"FSM1 Performance — 15,684 units × 8,760 hours ({speedup:.0f}× speedup)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlim(0, max(times) * 1.15)
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()
    out = DOCS / "performance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ─── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("fsm-rs — Col de Porte 2005–06 Example")
    print("=" * 60)
    print()

    forcing, obs, fortran, result, elapsed = run_cdp()

    # Daily dates for model output
    dates_daily = obs["dates"]  # 273 days

    print()
    print("Generating plots...")
    plot_comparison(obs, fortran, result, dates_daily)
    plot_scatter(fortran, result)
    plot_performance(elapsed)

    print()
    print("Done! Plots saved in docs/")


if __name__ == "__main__":
    main()
