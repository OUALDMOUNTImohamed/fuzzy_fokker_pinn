# -*- coding: utf-8 -*-
"""PyTorch PINN teacher and conservative ANN/ANFIS students for Colab.

No previously computed NumPy result is reused as a PyTorch training result.
Execute run_experiment(make_config("quick")) or run this script with --help.
"""
from __future__ import annotations
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
import argparse
import copy
import itertools
import json
import math
import random
import sys
import time
import platform
from dataclasses import dataclass, asdict
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime
import zipfile
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from scipy.special import gamma
from scipy.sparse.linalg import spsolve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch
from torch import nn
from torch.nn import functional as F

MODELS=("ANN","Gaussian","Bell","Sigmoid")
COLORS={"ANN":"#244f9c","Gaussian":"#198568","Bell":"#b66519","Sigmoid":"#a53a68"}
DISPLAY={"ANN":"ANN","Gaussian":"ANFIS Gaussian","Bell":"ANFIS Bell","Sigmoid":"ANFIS Sigmoid"}
LENGTH=2*np.pi
DMIN,DMAX,CORE=0.4,0.9,0.6
AMPLITUDE=0.6
TARGET_SCALE=0.1

def features(t, d, beta, horizon=1.0):
    return np.column_stack((2 * (np.asarray(t) / horizon) ** beta - 1,
                            2 * (np.asarray(d) - DMIN) / (DMAX - DMIN) - 1))

def basis(x, k=4):
    freq = np.arange(1, k + 1)
    return np.column_stack((np.cos(np.outer(x, freq)), np.sin(np.outer(x, freq))))

def initial_u(x):
    return 1 + 0.2 * np.cos(x)

def constrain_coefficients(c):
    """Projection radiale sur la boule l1 de rayon A.

    La carte est l'identite pour ||c||_1 <= A. Ce choix differe de la
    normalisation lisse du brouillon : on apprend ici les coefficients bornes
    c eux-memes, et non les variables latentes v du PINN.
    """
    radius = np.sum(np.abs(c), axis=-1, keepdims=True)
    return c / np.maximum(1.0, radius / AMPLITUDE)

def reconstruct(c, t, x, beta, horizon=1.0):
    c = constrain_coefficients(c)
    return (initial_u(x)[None, :] + (np.asarray(t) / horizon)[:, None] ** beta
            * (c @ basis(x, c.shape[1] // 2).T)) / LENGTH

def exact_coefficients(d, k=4):
    z = 2 * np.pi * (np.asarray(d) - DMIN) / (DMAX - DMIN)
    c = np.zeros((len(z), 2 * k))
    c[:, 1] = 0.04 * np.cos(z)  # cos(2x)
    c[:, k] = 0.08 * np.sin(z)  # sin(x)
    return c

def exact_density(x, t, d, beta, k=4):
    return reconstruct(exact_coefficients(d, k), t, x, beta)

def exact_cuts(x, t, r, beta):
    """Extrema analytiques en D, y compris les points stationnaires interieurs."""
    lo, hi = DMIN + 0.2 * r, DMAX - 0.3 * r
    zlo, zhi = 2 * np.pi * (np.array([lo, hi]) - DMIN) / (DMAX - DMIN)
    aa, bb = 0.08 * np.sin(x), 0.04 * np.cos(2 * x)
    phi = np.arctan2(aa, bb)
    candidates = [np.full_like(x, zlo), np.full_like(x, zhi)]
    for shift in range(-2, 3):
        for offset in (0.0, np.pi):
            z = phi + offset + 2 * np.pi * shift
            candidates.append(np.where((z >= zlo) & (z <= zhi), z, np.nan))
    zz = np.asarray(candidates)
    values = aa[None, :] * np.sin(zz) + bb[None, :] * np.cos(zz)
    base = initial_u(x)
    return ((base + t ** beta * np.nanmin(values, axis=0)) / LENGTH,
            (base + t ** beta * np.nanmax(values, axis=0)) / LENGTH)

def l1_matrix(times, beta):
    n = len(times) - 1
    b = np.zeros((n + 1, n + 1))
    for j in range(1, n + 1):
        a = ((times[j] - times[:j]) ** (1 - beta)
             - (times[j] - times[1:j + 1]) ** (1 - beta)) / (gamma(2 - beta) * np.diff(times[:j + 1]))
        b[j, :j] -= a
        b[j, 1:j + 1] += a
    return b

def solve_unforced(times, diffusions, beta, nx=64):
    """Solveur independant : L1 non uniforme, differences centrees, Newton."""
    x = np.arange(nx) * LENGTH / nx
    dx = LENGTH / nx
    ix = np.arange(nx)
    dxmat = sparse.coo_matrix((np.r_[np.full(nx, .5 / dx), np.full(nx, -.5 / dx)],
                             (np.r_[ix, ix], np.r_[(ix + 1) % nx, (ix - 1) % nx])), shape=(nx, nx)).tocsr()
    dxx = sparse.coo_matrix((np.r_[np.ones(nx), -2 * np.ones(nx), np.ones(nx)] / dx ** 2,
                            (np.r_[ix, ix, ix], np.r_[(ix - 1) % nx, ix, (ix + 1) % nx])), shape=(nx, nx)).tocsr()
    eye = sparse.eye(nx, format="csr")
    b = l1_matrix(times, beta)
    states = np.empty((len(diffusions), len(times), nx))
    worst = 0.0
    for j, d in enumerate(diffusions):
        states[j, 0] = initial_u(x)
        for n in range(1, len(times)):
            past = b[n, :n] @ states[j, :n]
            u = states[j, n - 1].copy()
            tol = max(2e-10, 2e-13 * b[n, n])
            for _ in range(25):
                residual = b[n, n] * u + past + dxmat @ u - d / LENGTH * (dxx @ (u * u))
                err = np.max(np.abs(residual))
                if err < tol:
                    break
                jac = b[n, n] * eye + dxmat - 2 * d / LENGTH * (dxx @ sparse.diags(u))
                step = spsolve(jac.tocsc(), residual)
                factor = 1.0
                for _ in range(20):
                    candidate = u - factor * step
                    rc = b[n, n] * candidate + past + dxmat @ candidate - d / LENGTH * (dxx @ candidate ** 2)
                    if candidate.min() > 0 and np.max(np.abs(rc)) < err:
                        u = candidate
                        break
                    factor *= 0.5
                else:
                    raise RuntimeError(f"Newton line search failed: D={d}, n={n}")
            else:
                raise RuntimeError(f"Newton failed: D={d}, n={n}")
            states[j, n] = u
            worst = max(worst, err)
        if (j + 1) % 10 == 0:
            print(f"  FD reference {j + 1}/{len(diffusions)}", flush=True)
    return x, states / LENGTH, {"max_newton_residual_scaled_u": worst,
                              "mass_defect_reference": float(np.max(np.abs(states.mean(axis=2) - 1))),
                              "minimum_reference_density": float(states.min() / LENGTH)}

def model_coeff(model, times, ds, beta):
    return constrain_coefficients(model.predict(features(times, ds, beta)) * TARGET_SCALE)

def evaluate_cuts(model, args, problem, xr=None, refs=None, ds_reference=None):
    x = np.linspace(0, LENGTH, 192, endpoint=False) if problem == "manufactured" else xr
    levels = [0., .25, .5, .75, 1.]
    master = np.unique(np.r_[np.linspace(DMIN, DMAX, 401),
                             [DMIN + .2 * r for r in levels], [DMAX - .3 * r for r in levels]])
    c = model_coeff(model, np.ones(len(master)), master, args.beta)
    p = reconstruct(c, np.ones(len(master)), x, args.beta)
    rows, bands, previous = [], {}, None
    for r in levels:
        low, high = DMIN + .2 * r, DMAX - .3 * r
        select = (master >= low - 1e-12) & (master <= high + 1e-12)
        lower, upper = p[select].min(axis=0), p[select].max(axis=0)
        if problem == "manufactured":
            rl, ru = exact_cuts(x, 1., r, args.beta)
            # Erreur de recherche seule, sur la solution exacte.
            ep = exact_density(x, np.ones(select.sum()), master[select], args.beta, args.k)
            search_error = max(np.max(np.abs(ep.min(axis=0) - rl)), np.max(np.abs(ep.max(axis=0) - ru)))
            mode = "analytic continuous-D cuts"
        else:
            # Comparaison sur les MEMES diffusions reference, sans pretendre
            # connaitre les extrema continus du probleme non force.
            sd = (ds_reference >= low - 1e-12) & (ds_reference <= high + 1e-12)
            dr = ds_reference[sd]
            pc = reconstruct(model_coeff(model, np.ones(len(dr)), dr, args.beta), np.ones(len(dr)), x, args.beta)
            rl, ru = refs[sd, -1].min(axis=0), refs[sd, -1].max(axis=0)
            lower_eval, upper_eval = pc.min(axis=0), pc.max(axis=0)
            search_error = np.nan
            mode = "same finite diffusion sample for model and FD reference"
        if problem == "manufactured":
            lower_eval, upper_eval = lower, upper
        nesting = 0. if previous is None else max(0., np.max(previous[0] - lower), np.max(upper - previous[1]))
        rows.append({"level": r, "Hausdorff_error": float(max(np.max(abs(lower_eval - rl)), np.max(abs(upper_eval - ru)))),
                     "width_error": float(np.max(abs((upper_eval - lower_eval) - (ru - rl)))),
                     "exact_search_only_error": search_error, "nesting_defect": float(nesting), "reference": mode})
        bands[r] = (lower, upper, rl, ru)
        previous = lower, upper
    return pd.DataFrame(rows), (x, master, p, bands)

def savefig(fig, folder, name):
    png = folder / f"{name}.partial.png"
    svg = folder / f"{name}.partial.svg"
    fig.savefig(png, dpi=190, bbox_inches="tight")
    # Verifie que le fichier a ete ecrit en entier avant de le publier.
    plt.imread(png)
    png.replace(folder / f"{name}.png")
    fig.savefig(svg, bbox_inches="tight")
    svg.replace(folder / f"{name}.svg")
    plt.close(fig)

def make_figures(out, data, models, args, problem, examples, cuts_data):
    folder = out / "figures"; folder.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.constrained_layout.use": True})
    x, te, de, reference = examples
    nx = len(x); budget = max(args.sizes)
    xx, dd = np.meshgrid(x, np.linspace(DMIN, DMAX, 101))
    if problem == "manufactured":
        density = exact_density(x, np.ones(len(dd)), dd[:, 0], args.beta, args.k)
    else:
        density = data["reference"][:, -1]
        dd, xx = np.meshgrid(data["ds"], x, indexing="ij")
    fig = plt.figure(figsize=(11, 4.2))
    ax = fig.add_subplot(121, projection="3d")
    ax.plot_surface(xx, dd, density, cmap="viridis", linewidth=0, antialiased=True)
    ax.set(xlabel="x", ylabel="D", zlabel="p(x,1;D)", title="Reference density, t=1")
    ax = fig.add_subplot(122)
    for d in [.4, .6, .9]:
        if problem == "manufactured":
            pp = exact_density(x, np.array([1.]), np.array([d]), args.beta, args.k)[0]
        else:
            pp = density[np.argmin(abs(data["ds"] - d))]
        ax.plot(x, pp, label=f"D={d}")
    ax.set(xlabel="x", ylabel="Density", title="Reference profiles"); ax.legend()
    savefig(fig, folder, "fig01_reference")

    predicted = {}
    for name, model in models.items():
        predicted[name] = reconstruct(model_coeff(model, te, de, args.beta), te, x, args.beta)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for col, name in enumerate(("ANN", "Gaussian")):
        h = models[name].history
        axes[0, col].semilogy(h[:, 0], np.maximum(h[:, 1], 1e-20), label="Training")
        axes[0, col].semilogy(h[:, 0], np.maximum(h[:, 2], 1e-20), label="Validation")
        axes[0, col].set(title=DISPLAY[name], xlabel="Optimization step", ylabel="Coefficient MSE")
        axes[0, col].legend()
        ref, pred = reference.ravel(), predicted[name].ravel()
        ix = np.linspace(0, len(ref) - 1, min(3500, len(ref)), dtype=int)
        axes[1, col].scatter(ref[ix], pred[ix], s=4, alpha=.3, color=COLORS[name])
        lim = [min(ref.min(), pred.min()), max(ref.max(), pred.max())]
        axes[1, col].plot(lim, lim, "k--", lw=1)
        axes[1, col].set(xlabel="Reference density", ylabel="Predicted density")
    savefig(fig, folder, "fig02_loss_parity")

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for col, name in enumerate(("Gaussian", "Bell", "Sigmoid")):
        h = models[name].history
        axes[0, col].semilogy(h[:, 0], np.maximum(h[:, 1], 1e-20), label="Training")
        axes[0, col].semilogy(h[:, 0], np.maximum(h[:, 2], 1e-20), label="Validation")
        axes[0, col].set(title=DISPLAY[name], xlabel="Optimization step", ylabel="Coefficient MSE")
        axes[0, col].legend()
        ref, pred = reference.ravel(), predicted[name].ravel()
        ix = np.linspace(0, len(ref) - 1, min(3000, len(ref)), dtype=int)
        axes[1, col].scatter(ref[ix], pred[ix], s=3, alpha=.25, color=COLORS[name])
        lim = [min(ref.min(), pred.min()), max(ref.max(), pred.max())]
        axes[1, col].plot(lim, lim, "k--", lw=1)
        axes[1, col].set(xlabel="Reference density", ylabel="Predicted density")
    savefig(fig, folder, "fig03_membership_comparison")

    # Cartes sur les diffusions de TEST seulement, a t=1.
    select = np.isclose(te, 1.)
    errors = {n: np.abs(p[select] - reference[select]) for n, p in predicted.items()}
    vmax = max(e.max() for e in errors.values()); vmin = max(vmax * 1e-5, 1e-12)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for ax, name in zip(axes.ravel(), MODELS):
        im = ax.pcolormesh(x, de[select], np.maximum(errors[name], vmin), shading="auto", cmap="magma",
                           norm=LogNorm(vmin=vmin, vmax=max(vmax, 10 * vmin)))
        ax.set(title=DISPLAY[name], xlabel="x", ylabel="Held-out D")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Absolute density error (common log scale)")
    savefig(fig, folder, "fig04_absolute_errors")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in MODELS:
        sub = data["metrics"][data["metrics"].model == name].groupby("N_train")
        for ax, key in zip(axes, ("RMSE", "MAE")):
            mu, sd = sub[key].mean(), sub[key].std().fillna(0)
            ax.errorbar(mu.index, mu.values, yerr=sd.values, marker="o", capsize=3,
                        label=DISPLAY[name], color=COLORS[name])
            ax.set(xlabel="Training coefficient vectors", ylabel=key, yscale="log")
    axes[0].legend(fontsize=8)
    savefig(fig, folder, "fig05_data_budget")

    grid = np.linspace(-1, 1, 401)
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True)
    for row, name in enumerate(("Gaussian", "Bell", "Sigmoid")):
        mu = models[name].memberships(np.column_stack((grid, grid)))
        for j in range(2):
            for q in range(models[name].q):
                axes[row, j].plot(grid, mu[:, j, q], label=f"MF {q + 1}")
            axes[row, j].set(title=f"{DISPLAY[name]}: {'fractional time' if j == 0 else 'diffusion'}",
                             xlabel="Normalized input", ylabel="Membership", ylim=(-.03, 1.03))
            axes[row, j].legend(fontsize=8)
    savefig(fig, folder, "fig06_learned_memberships")

    st, sd = np.meshgrid(np.linspace(-1, 1, 70), np.linspace(-1, 1, 70), indexing="ij")
    _, weights, _, _, _ = models["Gaussian"].forward(np.column_stack((st.ravel(), sd.ravel())), models["Gaussian"].theta, True)
    fig, axes = plt.subplots(args.q, args.q, figsize=(9, 8), squeeze=False)
    for r, ax in enumerate(axes.ravel()):
        im = ax.imshow(weights[:, r].reshape(st.shape), origin="lower", extent=[-1, 1, -1, 1],
                       aspect="auto", vmin=0, vmax=1, cmap="viridis")
        a, b = models["Gaussian"].rules[r]
        ax.set(title=f"Rule ({a + 1},{b + 1})", xlabel="Normalized D", ylabel="Normalized fractional time")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Normalized firing weight")
    savefig(fig, folder, "fig07_rule_activation")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    xx, master, pp, bands = cuts_data["Gaussian"]
    for r, color in zip((0., .5, 1.), ("#78bca7", "#317a92", "#312f70")):
        lo, hi, rl, ru = bands[r]
        axes[0].fill_between(xx, lo, hi, alpha=.22, color=color, label=f"ANFIS, r={r:g}")
        axes[0].plot(xx, rl, color=color, ls="--", lw=1)
        axes[0].plot(xx, ru, color=color, ls="--", lw=1)
        axes[1].plot(xx, hi - lo, color=color, label=f"ANFIS, r={r:g}")
        axes[1].plot(xx, ru - rl, color=color, ls="--", lw=1)
    axes[0].set(xlabel="x", ylabel="Density interval", title="Fuzzy cuts at t=1; reference dashed")
    axes[1].set(xlabel="x", ylabel="Upper minus lower", title="Uncertainty widths")
    for ax in axes:
        ax.legend(fontsize=8)
    savefig(fig, folder, "fig08_fuzzy_cuts")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    dense = np.linspace(DMIN, DMAX, 501)
    xp = np.array([np.pi / 2])
    for name in MODELS:
        pp = reconstruct(model_coeff(models[name], np.ones(len(dense)), dense, args.beta), np.ones(len(dense)), xp, args.beta)[:, 0]
        ax.plot(dense, pp, label=DISPLAY[name], color=COLORS[name])
    if problem == "manufactured":
        exact = exact_density(xp, np.ones(len(dense)), dense, args.beta, args.k)[:, 0]
        ax.plot(dense, exact, "k--", label="Exact", lw=2)
        ax.scatter([DMIN, DMAX], [exact[0], exact[-1]], color="black", s=35, zorder=5)
    else:
        j = np.argmin(abs(data["xr"] - np.pi / 2))
        ax.plot(data["ds"], data["reference"][:, -1, j], "k--", label="FD reference")
    ax.set(xlabel="Diffusion D", ylabel="p(pi/2,1;D)", title="Parameter dependence and interior extrema")
    ax.legend(fontsize=8)
    savefig(fig, folder, "fig09_parameter_extrema")

def latex_table(frame, path, columns, caption):
    # Export autonome : pandas.to_latex necessiterait Jinja2 sur certaines versions.
    labels = {"model": "Model", "parameters": "Parameters", "N_train": "$N$",
              "R2_mean": "$R^2$ mean", "RMSE_mean": "RMSE mean", "RMSE_std": "RMSE SD",
              "MAE_mean": "MAE mean", "Hausdorff_error": "$E_H$", "width_error": "$E_W$",
              "nesting_defect": "Nesting defect", "mass_defect_max": "Mass defect",
              "minimum_density": "$\\min P$", "perturbation_relative_mean": "Perturbation error",
              "train_seconds_mean": "Training (s)", "inference_seconds_mean": "Evaluation (s)"}
    text = "\\begin{table}[H]\n\\centering\\small\n\\caption{" + caption + "}\n"
    wide = len(columns) > 5
    if wide:
        text += "\\resizebox{\\linewidth}{!}{%\n"
    text += "\\begin{tabular}{" + "l" + "r" * (len(columns) - 1) + "}\n\\toprule\n"
    text += " & ".join(labels.get(c, c.replace("_", " ")) for c in columns) + " \\\\\n\\midrule\n"
    for values in frame[columns].itertuples(index=False, name=None):
        cells = []
        for value in values:
            if isinstance(value, (int, np.integer)):
                cells.append(str(value))
            elif isinstance(value, (float, np.floating)):
                cells.append(f"{value:.8f}" if columns[len(cells)] == "R2_mean" else f"{value:.3e}")
            else:
                cells.append(str(value).replace("_", r"\_"))
        text += " & ".join(cells) + " \\\\\n"
    text += "\\bottomrule\n\\end{tabular}" + ("}" if wide else "") + "\n\\end{table}\n"
    path.write_text(text, encoding="utf-8")

# PYTORCH IMPLEMENTATION

@dataclass
class ExperimentConfig:
    profile: str = "quick"
    problems: tuple = ("manufactured",)
    label_source: str = "pinn"  # "pinn" or "reference" (explicit oracle ablation)
    device: str = "auto"
    output: str = ""
    beta: float = 0.75
    horizon: float = 1.0
    k: int = 4
    q: int = 3
    label_nt: int = 64
    label_nd: int = 41
    reference_nx: int = 64
    reference_refinement: bool = True
    teacher_nt: int = 48
    teacher_nx: int = 24
    teacher_nd: int = 9
    teacher_width: int = 24
    teacher_seed: int = 2026
    teacher_adam: int = 1200
    teacher_lbfgs: int = 80
    student_adam: int = 500
    student_lbfgs: int = 80
    learning_rate: float = 0.003
    sizes: tuple = (100, 400)
    seeds: tuple = (11,)
    split_seed: int = 2026
    check_every: int = 50
    dtype: str = "float64"


def make_config(profile="quick", **overrides):
    cfg = ExperimentConfig(profile=profile)
    if profile == "research":
        cfg.problems = ("manufactured", "unforced")
        cfg.teacher_adam, cfg.teacher_lbfgs = 4000, 200
        cfg.student_adam, cfg.student_lbfgs = 1500, 200
        cfg.teacher_nt, cfg.teacher_nx, cfg.reference_nx = 64, 48, 128
        cfg.label_nt = 128
        cfg.sizes = (100, 200, 300, 400)
        cfg.seeds = (11, 23, 47, 71, 101)
    elif profile == "smoke":
        cfg.problems = ("manufactured", "unforced")
        cfg.teacher_adam, cfg.teacher_lbfgs = 25, 10
        cfg.student_adam, cfg.student_lbfgs = 25, 10
        cfg.teacher_nt, cfg.teacher_nx, cfg.teacher_nd = 12, 16, 5
        cfg.label_nt, cfg.label_nd, cfg.reference_nx = 24, 17, 32
        cfg.reference_refinement = False
        cfg.sizes, cfg.seeds, cfg.check_every = (32,), (11,), 10
    elif profile != "quick":
        raise ValueError("profile must be quick, research, or smoke")
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise ValueError(f"Unknown configuration key: {key}")
        setattr(cfg, key, value)
    if not cfg.output:
        parent = Path("/content") if Path("/content").exists() else Path.cwd()
        cfg.output = str(parent / ("fokker_torch_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    return cfg


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def setup_runtime(cfg):
    if cfg.dtype != "float64":
        raise ValueError("Use float64 for the graded fractional-memory operator")
    if not 0 < cfg.beta < 1 or cfg.horizon != 1.0:
        raise ValueError("This benchmark implements 0<beta<1 and T=1")
    if cfg.k < 2 or cfg.reference_nx <= 2 * cfg.k or cfg.teacher_nx <= 2 * cfg.k:
        raise ValueError("K>=2 and both spatial grids must have more than 2K points")
    if cfg.q < 2 or cfg.label_nd < 13 or cfg.label_nt < 4:
        raise ValueError("Require q>=2, label_nd>=13, label_nt>=4")
    if cfg.label_source not in ("pinn", "reference"):
        raise ValueError("label_source must be pinn or reference")
    if min(cfg.sizes) < 8 or min(cfg.teacher_adam, cfg.student_adam) < 0:
        raise ValueError("Invalid budget or epoch count")
    torch.set_default_dtype(torch.float64)
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; select device='auto' or 'cpu'")
    if device.type == "cpu":
        torch.set_num_threads(min(2, os.cpu_count() or 1))
    return device


def tensor(a, device):
    return torch.as_tensor(a, dtype=torch.float64, device=device)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def save_checkpoint(payload, path):
    path = Path(path)
    temporary = path.with_suffix(".partial.pt")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_trained_family(path, device="cpu"):
    """Reload checkpoints created by this program, using weights_only=True."""
    payload = torch.load(path, map_location=device, weights_only=True)
    cfg = ExperimentConfig(**payload["configuration"])
    kind = payload.get("model", "PINN")
    if kind in ("PINN", "ANN"):
        net = FeatureMLP(cfg.teacher_width if kind == "PINN" else 10, cfg.k)
    else:
        net = NeuroFuzzyTSK(kind, cfg.k, cfg.q)
    family = DensityFamily(net, cfg).to(device=device, dtype=torch.float64)
    family.load_state_dict(payload["state_dict"])
    family.eval()
    return family, cfg


class FeatureMLP(nn.Module):
    def __init__(self, width, k):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(2, width), nn.Tanh(),
                                    nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 2 * k))
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        # Small initial corrections; the density constraints hold for all weights.
        with torch.no_grad():
            self.layers[-1].weight.mul_(0.1)

    def forward(self, x):
        return self.layers(x)


class NeuroFuzzyTSK(nn.Module):
    """Shared antecedents and vector affine consequents. No grid-rule pruning."""
    def __init__(self, kind, k=4, q=3):
        super().__init__()
        self.kind, self.q, self.k = kind, q, k
        centers = torch.linspace(-0.8, 0.8, q).repeat(2, 1)
        self.centers = nn.Parameter(centers + 0.025 * torch.randn(2, q))
        self.scale = nn.Parameter(torch.full((2, q), math.log(math.expm1(0.62))))
        if kind == "Sigmoid":
            self.scale = nn.Parameter(torch.linspace(-4.0, 4.0, q).repeat(2, 1))
        if kind == "Bell":
            self.shape = nn.Parameter(torch.full((2, q), math.log(math.expm1(1.0))))
        self.register_buffer("rules", torch.tensor(list(itertools.product(range(q), repeat=2)), dtype=torch.long))
        self.consequents = nn.Parameter(0.05 * torch.randn(q * q, 3, 2 * k))

    def log_memberships(self, x):
        diff = x[:, :, None] - self.centers[None, :, :]
        if self.kind == "Sigmoid":
            return F.logsigmoid(self.scale[None, :, :] * diff)
        width = 0.03 + F.softplus(self.scale)
        z = diff / width[None, :, :]
        if self.kind == "Gaussian":
            return -0.5 * z.square()
        if self.kind == "Bell":
            exponent = 0.5 + F.softplus(self.shape)
            logabs = 0.5 * torch.log(z.square() + 1e-12)
            return -F.softplus(2 * exponent[None, :, :] * logabs)
        raise ValueError(self.kind)

    def firing_weights(self, x):
        logs = self.log_memberships(x)
        rule_logs = logs[:, 0, self.rules[:, 0]] + logs[:, 1, self.rules[:, 1]]
        return torch.softmax(rule_logs, dim=1)

    def forward(self, x):
        weights = self.firing_weights(x)
        xb = torch.cat((x, torch.ones_like(x[:, :1])), dim=1)
        local = torch.einsum("ni,rim->nrm", xb, self.consequents)
        return torch.einsum("nr,nrm->nm", weights, local)


class DensityFamily(nn.Module):
    """Smooth bounded Fourier head shared by teacher and students.

    This is NOT the radial coefficient clipping in the previous NumPy program.
    c(v)=A*v/(1+sum(sqrt(v_j^2+eta^2))) is smooth and sum|c_j|<A.
    """
    def __init__(self, coefficient_network, cfg):
        super().__init__()
        self.network = coefficient_network
        self.k, self.beta, self.horizon = cfg.k, cfg.beta, cfg.horizon
        self.amplitude, self.eta = AMPLITUDE, 1e-6

    def coefficients(self, normalized_inputs):
        raw = self.network(normalized_inputs)
        denominator = 1 + torch.sqrt(raw.square() + self.eta ** 2).sum(dim=-1, keepdim=True)
        return self.amplitude * raw / denominator

    def forward(self, normalized_inputs):
        return self.coefficients(normalized_inputs)


class FractionalContext:
    """Full L1 history at fixed diffusion, with analytic Fourier x derivatives."""
    def __init__(self, cfg, problem, times, diffusions, nx, device):
        self.cfg, self.problem, self.device = cfg, problem, device
        self.times, self.diffusions = np.asarray(times), np.asarray(diffusions)
        self.nd, self.nt = len(diffusions), len(times) - 1
        tt, dd = np.meshgrid(self.times, self.diffusions, indexing="xy")
        self.inputs = tensor(features(tt.ravel(), dd.ravel(), cfg.beta), device)
        self.b = tensor(l1_matrix(self.times, cfg.beta)[1:], device)
        self.g = tensor(self.times ** cfg.beta, device)[None, :, None]
        x = np.arange(nx) * LENGTH / nx
        freq = np.arange(1, cfg.k + 1)
        cos = np.cos(np.outer(x, freq)); sin = np.sin(np.outer(x, freq))
        self.phi = tensor(np.c_[cos, sin], device)
        self.phix = tensor(np.c_[-sin * freq, cos * freq], device)
        self.phixx = tensor(np.c_[-cos * freq ** 2, -sin * freq ** 2], device)
        self.u0 = tensor(initial_u(x), device)[None, None, :]
        self.u0x = tensor(-0.2 * np.sin(x), device)[None, None, :]
        self.u0xx = tensor(-0.2 * np.cos(x), device)[None, None, :]
        self.d = tensor(self.diffusions, device)[:, None, None]
        self.force_u = tensor(self.scaled_source(x), device)

    def scaled_source(self, x):
        if self.problem == "unforced":
            return np.zeros((self.nd, self.nt, len(x)))
        d = self.diffusions[:, None, None]
        tp = self.times[None, 1:, None] ** self.cfg.beta
        z = 2 * np.pi * (d - DMIN) / (DMAX - DMIN)
        a, b = 0.08 * np.sin(z), 0.04 * np.cos(z)
        xx = x[None, None, :]
        mode = a * np.sin(xx) + b * np.cos(2 * xx)
        u = 1 + 0.2 * np.cos(xx) + tp * mode
        ux = -0.2 * np.sin(xx) + tp * (a * np.cos(xx) - 2 * b * np.sin(2 * xx))
        uxx = -0.2 * np.cos(xx) + tp * (-a * np.sin(xx) - 4 * b * np.cos(2 * xx))
        return gamma(1 + self.cfg.beta) * mode + ux - 2 * d / LENGTH * (ux ** 2 + u * uxx)

    def residual(self, model):
        c = model.coefficients(self.inputs).reshape(self.nd, self.nt + 1, 2 * self.cfg.k)
        modes = self.g * c
        positive = modes[:, 1:, :]
        u = self.u0 + positive @ self.phi.T
        ux = self.u0x + positive @ self.phix.T
        uxx = self.u0xx + positive @ self.phixx.T
        # Apply L1 to corrections only: constants are annihilated analytically.
        memory_modes = torch.einsum("nk,dkm->dnm", self.b, modes)
        memory_u = memory_modes @ self.phi.T
        return (memory_u + ux - 2 * self.d / LENGTH * (ux.square() + u * uxx) - self.force_u) / LENGTH

    def loss(self, model):
        return self.residual(model).square().mean()


def fit_torch(model, train_loss, validation_loss, adam_steps, lbfgs_steps, cfg,
              name, out, history_scale=1.0):
    device = next(model.parameters()).device
    synchronize(device); started = time.perf_counter()
    best = {"value": math.inf, "state": None, "step": 0}
    rows = []

    def record(step, phase):
        model.eval()
        with torch.no_grad():
            tr = float(train_loss().detach().cpu())
            va = float(validation_loss().detach().cpu())
        if not np.isfinite(tr + va):
            raise FloatingPointError(f"Nonfinite loss: {name}, step={step}")
        rows.append({"step": int(step), "phase": phase, "train_loss": tr * history_scale,
                     "validation_loss": va * history_scale})
        if va < best["value"]:
            best.update(value=va, state=cpu_state(model), step=int(step))
        model.train()

    record(0, "initial")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, adam_steps), eta_min=cfg.learning_rate / 10)
    for step in range(1, adam_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        value = train_loss()
        if not torch.isfinite(value):
            raise FloatingPointError(f"Nonfinite Adam loss: {name}")
        value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step(); scheduler.step()
        if step % cfg.check_every == 0 or step == adam_steps:
            record(step, "Adam")
            if name == "PINN" and (step % max(100, cfg.check_every) == 0 or step == adam_steps):
                print(f"  PINN Adam {step}/{adam_steps}, val L1-MSE={rows[-1]['validation_loss']:.3e}", flush=True)

    model.load_state_dict(best["state"])
    closure_calls = 0
    if lbfgs_steps:
        optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=min(20, lbfgs_steps),
                                      history_size=40, tolerance_grad=1e-10, tolerance_change=1e-13,
                                      line_search_fn="strong_wolfe")
        first_parameter = next(model.parameters())
        previous = 0
        while previous < lbfgs_steps:
            optimizer.param_groups[0]["max_iter"] = min(20, lbfgs_steps - previous)

            def closure():
                nonlocal closure_calls
                closure_calls += 1
                optimizer.zero_grad(set_to_none=True)
                value = train_loss()
                if not torch.isfinite(value):
                    raise FloatingPointError(f"Nonfinite L-BFGS loss: {name}")
                value.backward()
                return value

            optimizer.step(closure)
            current = int(optimizer.state[first_parameter].get("n_iter", 0))
            record(adam_steps + current, "L-BFGS")
            if current <= previous:
                break
            previous = current
    model.load_state_dict(best["state"])
    model.eval()
    synchronize(device); elapsed = time.perf_counter() - started
    history = pd.DataFrame(rows)
    history.to_csv(out, index=False)
    return history, {"train_seconds": elapsed, "selected_step": best["step"],
                     "lbfgs_closure_evaluations": closure_calls,
                     "best_validation_loss": best["value"] * history_scale}


def predict_coefficients(model, normalized_inputs, batch_size=8192):
    device = next(model.parameters()).device
    predictions = []
    with torch.no_grad():
        for i in range(0, len(normalized_inputs), batch_size):
            predictions.append(model.coefficients(tensor(normalized_inputs[i:i + batch_size], device)).cpu().numpy())
    return np.concatenate(predictions, axis=0)


class PlotAdapter:
    """Expose physical coefficients to the independent NumPy evaluation tools."""
    def __init__(self, model, history, cfg):
        self.model, self.q = model, cfg.q
        self.theta = None
        self.history = history[["step", "train_loss", "validation_loss"]].to_numpy()
        self.rules = np.array(list(itertools.product(range(cfg.q), repeat=2)))

    def predict(self, x, theta=None):
        return predict_coefficients(self.model, x) / TARGET_SCALE

    def memberships(self, x):
        device = next(self.model.parameters()).device
        with torch.no_grad():
            return self.model.network.log_memberships(tensor(x, device)).exp().cpu().numpy()

    def forward(self, x, theta=None, derivatives=False):
        device = next(self.model.parameters()).device
        with torch.no_grad():
            weights = self.model.network.firing_weights(tensor(x, device)).cpu().numpy()
        return self.predict(x), weights, None, None, None


def density_metrics(predicted, reference, x):
    error = predicted - reference
    ss = np.sum((reference - reference.mean()) ** 2)
    baseline = initial_u(x)[None, :] / LENGTH
    return {"R2": float(1 - np.sum(error ** 2) / ss),
            "RMSE": float(np.sqrt(np.mean(error ** 2))), "MAE": float(np.mean(abs(error))),
            "E_inf": float(np.max(abs(error))),
            "relative_L2": float(np.linalg.norm(error) / np.linalg.norm(reference)),
            "perturbation_relative_error": float(np.linalg.norm(error) / max(np.linalg.norm(reference - baseline), 1e-15)),
            "mass_defect": float(np.max(abs(LENGTH * predicted.mean(axis=1) - 1))),
            "minimum_density": float(predicted.min())}


def verify_implementation(cfg, device):
    seed_all(1729)
    x = tensor(np.random.default_rng(0).uniform(-0.8, 0.8, (7, 2)), device)
    checks = {}
    for name in MODELS:
        net = FeatureMLP(10, cfg.k) if name == "ANN" else NeuroFuzzyTSK(name, cfg.k, cfg.q)
        model = DensityFamily(net, cfg).to(device=device, dtype=torch.float64)
        parameters = list(model.parameters())
        value = model.coefficients(x).square().sum()
        gradients = torch.autograd.grad(value, parameters)
        maximum = 0.0
        for parameter, gradient in zip(parameters, gradients):
            for j in {0, parameter.numel() // 2, parameter.numel() - 1}:
                with torch.no_grad():
                    original = parameter.view(-1)[j].item()
                    parameter.view(-1)[j] = original + 1e-6
                    plus = model.coefficients(x).square().sum().item()
                    parameter.view(-1)[j] = original - 1e-6
                    minus = model.coefficients(x).square().sum().item()
                    parameter.view(-1)[j] = original
                maximum = max(maximum, abs((plus - minus) / 2e-6 - gradient.reshape(-1)[j].item()))
        if maximum > 1e-7:
            raise AssertionError(f"Gradient discrepancy for {name}: {maximum}")
        c = predict_coefficients(model, np.array([[0., -1.], [0.5, 0.], [1., 1.]]))
        xx = np.linspace(0, LENGTH, 64, endpoint=False)
        pp = reconstruct(c, np.array([0., .5, 1.]), xx, cfg.beta)
        if np.max(abs(pp.mean(axis=1) * LENGTH - 1)) > 1e-12 or pp.min() < (0.8 - AMPLITUDE) / LENGTH - 1e-12:
            raise AssertionError("Density invariants failed")
        checks[name] = {"max_gradient_absolute_error": maximum,
                        "parameters": sum(p.numel() for p in parameters)}
    tt = np.linspace(0, 1, 17) ** 3
    l1_error = np.max(abs(l1_matrix(tt, cfg.beta)[1:] @ tt - tt[1:] ** (1 - cfg.beta) / gamma(2 - cfg.beta)))
    if l1_error > 1e-9:
        raise AssertionError(f"L1 linear-function check failed: {l1_error}")
    checks["L1_linear_function_error"] = float(l1_error)
    return checks


def prepare_reference(cfg, problem):
    started = time.perf_counter()
    times = np.linspace(0, 1, cfg.label_nt + 1) ** 3
    ds = np.linspace(DMIN, DMAX, cfg.label_nd)
    ds[np.argmin(abs(ds - CORE))] = CORE
    ds.sort()
    if problem == "manufactured":
        x = np.linspace(0, LENGTH, max(128, cfg.reference_nx), endpoint=False)
        coeff = np.repeat(exact_coefficients(ds, cfg.k)[:, None, :], len(times), axis=1)
        dd, tt = np.meshgrid(ds, times, indexing="ij")
        reference = exact_density(x, tt.ravel(), dd.ravel(), cfg.beta, cfg.k).reshape(len(ds), len(times), len(x))
        diag = {"reference": "analytic manufactured solution", "coefficient_time_dependence_after_g": "constant"}
    elif problem == "unforced":
        x, reference, diag = solve_unforced(times, ds, cfg.beta, cfg.reference_nx)
        delta = LENGTH * reference - initial_u(x)[None, None, :]
        fft = np.fft.rfft(delta, axis=2) / len(x)
        coeff = np.concatenate((2 * fft[:, :, 1:cfg.k + 1].real,
                                -2 * fft[:, :, 1:cfg.k + 1].imag), axis=2)
        coeff[:, 1:] /= times[None, 1:, None] ** cfg.beta
        coeff[:, 0] = coeff[:, 1]
        diag["reference"] = "L1 centered finite differences; not an exact solution"
        if cfg.reference_refinement:
            _, fine, _ = solve_unforced(np.linspace(0, 1, 2 * cfg.label_nt + 1) ** 3,
                                        ds[[0, len(ds) // 2, -1]], cfg.beta, 2 * cfg.reference_nx)
            diag["joint_refinement_difference"] = float(np.max(abs(fine[:, ::2, ::2] - reference[[0, len(ds) // 2, -1]])))
    else:
        raise ValueError(problem)
    diag["reference_generation_seconds"] = time.perf_counter() - started
    return times, ds, x, reference, coeff, diag


def export_numerical_text(out, cfg, problem, diagnostics):
    text = (
        "\\subsection{PyTorch numerical experiments}\n"
        "A conservative parametric PINN is trained using a graded L1 history operator "
        "and analytic Fourier spatial derivatives. All realizations share a globally constant diffusion. "
        "The ANN and three ANFIS students use the same smooth positive, periodic, unit-mass density head. "
        "Its normalization is $c(v)=Av/[1+\\sum_j\\sqrt{v_j^2+\\eta^2}]$. "
        "This differs from radial coefficient clipping in the earlier NumPy implementation.\n\n"
    )
    text += ("Student labels are generated by the trained PINN. " if cfg.label_source == "pinn"
             else "This run is an oracle ablation: student labels are reference coefficients, not PINN outputs. ")
    text += (f"The problem is {problem}, with $\\beta={cfg.beta:g}$. "
             f"The training budgets are {', '.join(map(str,cfg.sizes))} coefficient vectors, "
             f"and {len(cfg.seeds)} student initializations are used. "
             "Complete diffusion values are held out for validation and testing, including from teacher collocation. "
             "Optimization combines Adam and L-BFGS; the retained iterate minimizes a validation objective. "
             "The teacher validation objective is a discrete L1 residual, whereas student validation uses coefficient-label MSE. "
             "All reported residuals remain discretization-dependent diagnostics, not continuous certificates.\n\n")
    if problem == "manufactured":
        text += ("The independent reference is the manufactured analytic density. "
                 "Its prescribed forcing depends jointly on diffusion. After extracting $t^\\beta$, "
                 "its exact coefficients are time-independent; this example alone does not establish "
                 "learning of nontrivial temporal coefficient dynamics.\n\n")
    else:
        text += "The independent reference uses conservative periodic L1 finite differences and Newton iteration. "
        if "joint_refinement_difference" in diagnostics:
            text += f"The observed joint space/time refinement difference at three diffusivities is ${diagnostics['joint_refinement_difference']:.3e}$; it is not a rigorous error bound. "
        text += "Unforced fuzzy errors compare predictions and reference on the same finite diffusion sample.\n\n"
    text += ("RMSE against the teacher measures student fidelity; RMSE against the independent reference "
             "measures total observed error. A high global $R^2$ can be influenced by the nonzero density background, "
             "so perturbation-relative errors are also exported. Zero mass defects follow from construction. "
             "Learned memberships and rule activations are descriptive and do not establish physical causality. "
             "The canonical fuzzy plots use the first prespecified seed at the largest budget. "
             "No universal neural or neuro-fuzzy superiority, PDE well-posedness, or optimizer convergence is claimed.\n")
    (out / "numerical_methods.tex").write_text(text, encoding="utf-8")


def run_problem(cfg, problem, device, output):
    out = output / problem
    for directory in (out, out / "models", out / "histories", out / "tables"):
        directory.mkdir(parents=True, exist_ok=True)
    times, ds, x, ref, oracle_coeff, diagnostics = prepare_reference(cfg, problem)
    val_ix = np.arange(3, len(ds) - 1, 6)
    test_ix = np.arange(1, len(ds) - 1, 6)
    train_ix = np.setdiff1d(np.arange(len(ds)), np.r_[val_ix, test_ix])
    teacher_d = ds[train_ix[np.linspace(0, len(train_ix) - 1, min(cfg.teacher_nd, len(train_ix)), dtype=int)]]
    seed_all(cfg.teacher_seed)
    teacher = DensityFamily(FeatureMLP(cfg.teacher_width, cfg.k), cfg).to(device)
    train_context = FractionalContext(cfg, problem, np.linspace(0, 1, cfg.teacher_nt + 1) ** 3,
                                      teacher_d, cfg.teacher_nx, device)
    validation_context = FractionalContext(cfg, problem, np.linspace(0, 1, 2 * cfg.teacher_nt + 1) ** 3,
                                           ds[val_ix], 2 * cfg.teacher_nx, device)
    print(f"\n{problem}: training PINN teacher on {device}", flush=True)
    teacher_history, teacher_info = fit_torch(teacher, lambda: train_context.loss(teacher),
                                             lambda: validation_context.loss(teacher),
                                             cfg.teacher_adam, cfg.teacher_lbfgs, cfg, "PINN",
                                             out / "histories" / "PINN.csv")
    save_checkpoint({"state_dict": cpu_state(teacher), "configuration": asdict(cfg), "problem": problem}, out / "models" / "PINN.pt")
    td, tt = np.meshgrid(ds, times[1:], indexing="ij")
    inputs_all = features(tt.ravel(), td.ravel(), cfg.beta)
    synchronize(device); started = time.perf_counter()
    teacher_labels = predict_coefficients(teacher, inputs_all).reshape(len(ds), len(times) - 1, 2 * cfg.k)
    synchronize(device); diagnostics["teacher_label_generation_seconds"] = time.perf_counter() - started
    labels = teacher_labels if cfg.label_source == "pinn" else oracle_coeff[:, 1:]
    np.savez_compressed(out / "data.npz", times=times, diffusions=ds, x=x, density_reference=ref,
                        reference_coefficients=oracle_coeff, teacher_coefficients=teacher_labels,
                        training_labels=labels, train_indices=train_ix, validation_indices=val_ix, test_indices=test_ix)

    def label_subset(indices):
        dd, t = np.meshgrid(ds[indices], times[1:], indexing="ij")
        return features(t.ravel(), dd.ravel(), cfg.beta), labels[indices].reshape(-1, 2 * cfg.k)

    xtrain, ytrain = label_subset(train_ix)
    xv, yv = label_subset(val_ix)
    if max(cfg.sizes) > len(xtrain):
        raise ValueError(f"Only {len(xtrain)} training vectors are available")
    order = np.random.default_rng(cfg.split_seed).permutation(len(xtrain))
    ti = np.arange(0, len(times), 2)
    if ti[-1] != len(times) - 1:
        ti = np.r_[ti, len(times) - 1]
    de, te = np.meshgrid(ds[test_ix], times[ti], indexing="ij")
    de, te = de.ravel(), te.ravel()
    eval_features = features(te, de, cfg.beta)
    truth = ref[test_ix][:, ti].reshape(-1, len(x))
    teacher_pred = reconstruct(predict_coefficients(teacher, eval_features), te, x, cfg.beta)
    teacher_metrics = {"model": "PINN", "parameters": sum(p.numel() for p in teacher.parameters()),
                       **density_metrics(teacher_pred, truth, x), **teacher_info}
    baselines = [teacher_metrics]
    oracle_pred = reconstruct(oracle_coeff[test_ix][:, ti].reshape(-1, 2 * cfg.k), te, x, cfg.beta)
    baselines.append({"model": "Fourier reference floor", "parameters": 0,
                      **density_metrics(oracle_pred, truth, x), "train_seconds": 0.0})
    # A simple interpolation comparator is useful for this two-dimensional input.
    from scipy.interpolate import RBFInterpolator
    started = time.perf_counter()
    rbf = RBFInterpolator(xtrain[order[:max(cfg.sizes)]], ytrain[order[:max(cfg.sizes)]], smoothing=1e-7)
    rbf_time = time.perf_counter() - started
    rbf_pred = reconstruct(rbf(eval_features), te, x, cfg.beta)
    baselines.append({"model": "RBF coefficient interpolation", "parameters": np.nan,
                      **density_metrics(rbf_pred, truth, x), "train_seconds": rbf_time})
    pd.DataFrame(baselines).to_csv(out / "tables" / "baselines.csv", index=False)

    rows, canonical_models, canonical_adapters, canonical_histories = [], {}, {}, {}
    xv_t, yv_t = tensor(xv, device), tensor(yv, device)
    for seed in cfg.seeds:
        for count in cfg.sizes:
            xx = tensor(xtrain[order[:count]], device)
            yy = tensor(ytrain[order[:count]], device)
            for name in MODELS:
                seed_all(seed)
                net = FeatureMLP(10, cfg.k) if name == "ANN" else NeuroFuzzyTSK(name, cfg.k, cfg.q)
                model = DensityFamily(net, cfg).to(device)
                stem = f"{name}_N{count}_seed{seed}"
                objective = lambda: ((model.coefficients(xx) - yy) / TARGET_SCALE).square().mean()
                validation = lambda: ((model.coefficients(xv_t) - yv_t) / TARGET_SCALE).square().mean()
                history, fit_info = fit_torch(model, objective, validation,
                                               cfg.student_adam, cfg.student_lbfgs, cfg, name,
                                               out / "histories" / f"{stem}.csv", TARGET_SCALE ** 2)
                synchronize(device); started = time.perf_counter()
                predicted = reconstruct(predict_coefficients(model, eval_features), te, x, cfg.beta)
                synchronize(device); eval_seconds = time.perf_counter() - started
                row = {"model": name, "N_train": count, "seed": seed,
                       "parameters": sum(p.numel() for p in model.parameters()),
                       **density_metrics(predicted, truth, x), **fit_info,
                       "RMSE_to_teacher": float(np.sqrt(np.mean((predicted - teacher_pred) ** 2))),
                       "inference_seconds": eval_seconds, "evaluation_density_values": predicted.size}
                rows.append(row)
                save_checkpoint({"state_dict": cpu_state(model), "configuration": asdict(cfg),
                                 "model": name, "problem": problem}, out / "models" / f"{stem}.pt")
                pd.DataFrame(rows).to_csv(out / "tables" / "runs.csv", index=False)
                print(f"  {name:8s} N={count:4d} seed={seed:3d} RMSE(ref)={row['RMSE']:.3e} "
                      f"RMSE(teacher)={row['RMSE_to_teacher']:.3e}", flush=True)
                if seed == cfg.seeds[0] and count == max(cfg.sizes):
                    canonical_models[name] = model
                    canonical_histories[name] = history
                    canonical_adapters[name] = PlotAdapter(model, history, cfg)

    frame = pd.DataFrame(rows)
    summary = frame.groupby(["model", "N_train"], sort=False).agg(
        parameters=("parameters", "first"), runs=("seed", "count"),
        R2_mean=("R2", "mean"), RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
        MAE_mean=("MAE", "mean"), fidelity_RMSE_mean=("RMSE_to_teacher", "mean"),
        train_seconds_mean=("train_seconds", "mean"), inference_seconds_mean=("inference_seconds", "mean")
    ).reset_index()
    summary.to_csv(out / "tables" / "summary.csv", index=False)
    args = SimpleNamespace(beta=cfg.beta, k=cfg.k, q=cfg.q, sizes=cfg.sizes)
    fuzzy_rows, cuts = [], {}
    for name, adapter in canonical_adapters.items():
        fuzzy, cuts[name] = evaluate_cuts(adapter, args, problem, x, ref, ds)
        fuzzy.insert(0, "model", name)
        fuzzy_rows.append(fuzzy)
    fuzzy = pd.concat(fuzzy_rows, ignore_index=True)
    fuzzy.to_csv(out / "tables" / "fuzzy_cuts.csv", index=False)
    residual_rows = []
    test_context = FractionalContext(cfg, problem, np.linspace(0, 1, 2 * cfg.teacher_nt + 1) ** 3,
                                     ds[test_ix], 2 * cfg.teacher_nx, device)
    for name, model in {"PINN": teacher, **canonical_models}.items():
        with torch.no_grad():
            rr = test_context.residual(model)
            residual_rows.append({"model": name, "fine_L1_residual_RMS": float(rr.square().mean().sqrt().cpu()),
                                  "fine_L1_residual_max": float(rr.abs().max().cpu())})
    pd.DataFrame(residual_rows).to_csv(out / "tables" / "discrete_residuals.csv", index=False)
    physics = frame[frame.N_train == max(cfg.sizes)].groupby("model", sort=False).agg(
        mass_defect_max=("mass_defect", "max"), minimum_density=("minimum_density", "min"),
        perturbation_relative_mean=("perturbation_relative_error", "mean")).reset_index()
    physics.to_csv(out / "tables" / "physics.csv", index=False)
    main = summary[summary.N_train == max(cfg.sizes)]
    latex_table(main, out / "tables" / "table_main.tex",
                ["model", "parameters", "R2_mean", "RMSE_mean", "RMSE_std", "MAE_mean"],
                "Student density errors against the independent reference; means and standard deviations.")
    latex_table(summary, out / "tables" / "table_budget.tex",
                ["model", "N_train", "parameters", "R2_mean", "RMSE_mean", "MAE_mean"],
                "Student data-budget study; counts refer to coefficient vectors.")
    latex_table(fuzzy[fuzzy.level == 0], out / "tables" / "table_fuzzy.tex",
                ["model", "Hausdorff_error", "width_error", "nesting_defect"],
                "Full-support fuzzy errors; canonical first seed at the largest budget.")
    latex_table(physics, out / "tables" / "table_physics.tex",
                ["model", "mass_defect_max", "minimum_density", "perturbation_relative_mean"],
                "Physical diagnostics at the largest budget.")
    latex_table(main, out / "tables" / "table_timing.tex",
                ["model", "train_seconds_mean", "inference_seconds_mean"],
                "Measured student training and evaluation costs; teacher and data costs are separate.")
    make_figures(out, {"metrics": frame, "reference": ref, "ds": ds, "xr": x}, canonical_adapters,
                 args, problem, (x, te, de, truth), cuts)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(teacher_history.step, teacher_history.train_loss.clip(lower=1e-20), label="Training L1 residual")
    ax.semilogy(teacher_history.step, teacher_history.validation_loss.clip(lower=1e-20), label="Validation L1 residual")
    ax.axvline(cfg.teacher_adam, color="gray", ls="--", label="Adam / L-BFGS switch")
    ax.set(xlabel="Optimization step", ylabel="Discrete residual MSE", title="PINN teacher")
    ax.legend(); savefig(fig, out / "figures", "fig10_pinn_teacher")
    export_numerical_text(out, cfg, problem, diagnostics)
    manifest = {"configuration": asdict(cfg), "problem": problem, "device": str(device),
                "torch_version": torch.__version__, "numpy_version": np.__version__,
                "scipy_version": scipy.__version__, "python_version": sys.version,
                "platform": platform.platform(), "teacher_metrics": teacher_metrics,
                "diagnostics": diagnostics, "student_fits_executed": len(frame),
                "teacher_train_diffusions": teacher_d.tolist(), "test_diffusions": ds[test_ix].tolist(),
                "density_head": "smooth bounded Fourier coefficients; not radial clipping",
                "warning": "Discrete residuals and reference refinement are not continuum certificates."}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"problem": problem, "output": str(out), "teacher": teacher_metrics,
            "summary": summary, "fits": len(frame)}


def package_results(output):
    output = Path(output)
    target = output.parent / (output.name + ".zip")
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(output.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                z.write(path, path.relative_to(output))
    return target


def run_experiment(cfg):
    device = setup_runtime(cfg)
    output = Path(cfg.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "configuration.json").write_text(json.dumps(asdict(cfg), indent=2))
    print(f"PyTorch {torch.__version__}; device={device}; dtype=float64; labels={cfg.label_source}", flush=True)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(device), flush=True)
    checks = verify_implementation(cfg, device)
    (output / "implementation_checks.json").write_text(json.dumps(checks, indent=2))
    results = []
    for problem in cfg.problems:
        results.append(run_problem(cfg, problem, device, output))
    try:
        source = Path(__file__)
        if source.is_file() and source.suffix == ".py":
            (output / "fokker_pytorch_colab.py").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    except NameError:
        pass  # Definitions executed directly in a notebook; notebook is the source.
    archive = package_results(output)
    print("Results:", output, "\nZIP:", archive, flush=True)
    return results, archive


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "quick", "research"), default="quick")
    parser.add_argument("--problem", choices=("manufactured", "unforced", "both"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="")
    parser.add_argument("--label-source", choices=("pinn", "reference"), default="pinn")
    args = parser.parse_args()
    cfg = make_config(args.profile, device=args.device, label_source=args.label_source)
    if args.output:
        cfg.output = args.output
    if args.problem:
        cfg.problems = ("manufactured", "unforced") if args.problem == "both" else (args.problem,)
    run_experiment(cfg)


if __name__ == "__main__":
    cli()
