import os
import re
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.optimize import least_squares, minimize_scalar

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


# =============================================================================
# Exponential Models & Robust Fitting
# =============================================================================

def exp_3param(x, A, B, C):
    return A * np.exp(np.clip(B * x, -500, 500)) + C


def exp_2param(x, A, B):
    return A * np.exp(np.clip(B * x, -500, 500))


def exp_decay_tau(x, A, tau, C):
    if tau == 0:
        return np.full_like(x, np.inf, dtype=float)
    return A * np.exp(np.clip(-x / tau, -500, 500)) + C


MODELS = {
    "y = A * exp(B * x) + C": {
        "func": exp_3param,
        "params": ["A", "B", "C"],
        "latex_eq": r"$y = A e^{B x} + C$",
    },
    "y = A * exp(B * x)": {
        "func": exp_2param,
        "params": ["A", "B"],
        "latex_eq": r"$y = A e^{B x}$",
    },
    "y = A * exp(-x / tau) + C": {
        "func": exp_decay_tau,
        "params": ["A", "tau", "C"],
        "latex_eq": r"$y = A e^{-x / \tau} + C$",
    },
}


def _log_linear_guess(x, y_shifted):
    """Estimate A, B for y_shifted ~ A * exp(B * x) using log-linear regression."""
    pos_mask = y_shifted > 0
    neg_mask = y_shifted < 0

    if np.sum(pos_mask) >= np.sum(neg_mask) and np.sum(pos_mask) >= 2:
        mask, sign = pos_mask, 1.0
    elif np.sum(neg_mask) >= 2:
        mask, sign = neg_mask, -1.0
    else:
        return None

    xm = x[mask]
    ym = np.log(np.abs(y_shifted[mask]))
    try:
        B0, lnA0 = np.polyfit(xm, ym, 1)
        A0 = sign * np.exp(np.clip(lnA0, -100, 100))
        return A0, B0
    except Exception:
        return None


def generate_initial_guesses(x, y, model_name):
    """Generate multiple candidate initial parameter guesses for curve_fit."""
    x_span = float(np.max(x) - np.min(x)) or 1.0
    y_min, y_max = float(np.min(y)), float(np.max(y))
    y_span = (y_max - y_min) or 1.0

    guesses = []

    if model_name == "y = A * exp(B * x)":
        g = _log_linear_guess(x, y)
        if g is not None:
            guesses.append(g)
        guesses.extend([
            (y[0] if y[0] != 0 else 1.0, -1.0 / x_span),
            (y[0] if y[0] != 0 else 1.0, 1.0 / x_span),
        ])
        return guesses

    # Candidate asymptotes C for 3-parameter models
    c_candidates = [
        y_min - 0.05 * y_span,
        y_max + 0.05 * y_span,
        0.0,
        float(y[-1]) - 0.01 * y_span,
        float(y[0]) - 0.01 * y_span,
    ]

    for C0 in c_candidates:
        g = _log_linear_guess(x, y - C0)
        if g is not None:
            A0, B0 = g
            if model_name == "y = A * exp(B * x) + C":
                guesses.append((A0, B0, C0))
            else:
                tau0 = -1.0 / B0 if abs(B0) > 1e-12 else x_span
                guesses.append((A0, tau0, C0))

    # Generic fallbacks
    if model_name == "y = A * exp(B * x) + C":
        guesses.extend([
            (y_span, -1.0 / x_span, y_min),
            (y_span, 1.0 / x_span, y_min),
            (-y_span, -1.0 / x_span, y_max),
        ])
    else:
        guesses.extend([
            (y_span, x_span, y_min),
            (y_span, -x_span, y_min),
            (-y_span, x_span, y_max),
        ])

    return guesses


def perform_exponential_fit(
    x, y, y_err, model_name, loss="linear", relative_weights=False
):
    """
    Globally-initialized exponential fit using:
      1. Coordinate normalization (u in [0, 1], v scaled to O(1))
      2. 1D Variable Projection (exact linear solve for A, C given b)
      3. High-precision nonlinear polish via scipy.optimize.least_squares

    Options:
      loss: "linear" (standard) or "soft_l1" / "cauchy" (outlier-resistant)
      relative_weights: if True (and no y_err provided), weights points by 1/|y|
                        so decaying tails across orders of magnitude fit accurately.
    """
    spec = MODELS[model_name]
    func = spec["func"]
    param_names = spec["params"]
    has_c = len(param_names) == 3

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    sigma = y_err[order] if y_err is not None else None

    if len(x) <= len(param_names):
        raise ValueError(f"Need more than {len(param_names)} valid data points (got {len(x)}).")

    # 1. Normalize coordinates to prevent ill-conditioning
    x0 = float(x[0])
    s_x = float(x[-1] - x[0]) or 1.0
    s_y = float(np.max(np.abs(y))) or 1.0

    u = (x - x0) / s_x
    v = y / s_y

    if sigma is not None:
        w = s_y / sigma
    elif relative_weights:
        w = 1.0 / np.maximum(np.abs(v), 1e-3)
    else:
        w = np.ones_like(v)

    # 2. Variable Projection: for a given b, solve linear least squares for (a, c)
    def solve_linear(b):
        eb = np.exp(np.clip(b * u, -500, 500))
        if has_c:
            M = np.column_stack([eb, np.ones_like(u)])
            coef, *_ = np.linalg.lstsq(M * w[:, None], v * w, rcond=None)
            a, c = float(coef[0]), float(coef[1])
            pred = a * eb + c
            return (a, c), float(np.sum(((v - pred) * w) ** 2))
        else:
            eb_w = eb * w
            denom = float(np.dot(eb_w, eb_w))
            a = float(np.dot(eb_w, v * w) / denom) if denom > 0 else 1.0
            pred = a * eb
            return (a,), float(np.sum(((v - pred) * w) ** 2))

    # Coarse 1D global grid search over b (covers flat to ultra-steep exponentials)
    b_pos = np.geomspace(1e-3, 50.0, 150)
    b_grid = np.concatenate([-b_pos[::-1], [0.0], b_pos])
    costs = [solve_linear(b)[1] for b in b_grid]
    i_best = int(np.argmin(costs))

    # Fine 1D refinement around best grid point
    lo = b_grid[max(0, i_best - 2)]
    hi = b_grid[min(len(b_grid) - 1, i_best + 2)]
    opt_1d = minimize_scalar(lambda b: solve_linear(b)[1], bounds=(lo, hi), method="bounded")
    b_init = float(opt_1d.x)
    lin_init, _ = solve_linear(b_init)

    p0_norm = np.array([lin_init[0], b_init, lin_init[1]]) if has_c else np.array([lin_init[0], b_init])

    # 3. High-precision nonlinear polish in normalized space
    def residuals_norm(p):
        if has_c:
            pred = p[0] * np.exp(np.clip(p[1] * u, -500, 500)) + p[2]
        else:
            pred = p[0] * np.exp(np.clip(p[1] * u, -500, 500))
        return (pred - v) * w

    res = least_squares(
        residuals_norm,
        p0_norm,
        method="trf",
        loss=loss,
        ftol=1e-13,
        xtol=1e-13,
        gtol=1e-13,
        max_nfev=50000,
    )
    p_norm = res.x

    # Compute normalized covariance matrix
    dof = len(x) - len(p_norm)
    JTJ_inv = np.linalg.pinv(res.jac.T @ res.jac)
    if sigma is not None:
        pcov_norm = JTJ_inv
    else:
        s_sq = float(np.sum(res.fun ** 2) / dof) if dof > 0 else np.nan
        pcov_norm = JTJ_inv * s_sq

    # 4. Transform parameters & covariance back to original units
    a_n, b_n = p_norm[0], p_norm[1]
    B = b_n / s_x
    shift_factor = np.exp(np.clip(-B * x0, -700, 700))
    A = s_y * a_n * shift_factor

    if model_name == "y = A * exp(B * x) + C":
        C = s_y * p_norm[2]
        popt = np.array([A, B, C])
        T = np.array([
            [s_y * shift_factor, -A * x0 / s_x, 0.0],
            [0.0,                1.0 / s_x,     0.0],
            [0.0,                0.0,           s_y],
        ])
    elif model_name == "y = A * exp(B * x)":
        popt = np.array([A, B])
        T = np.array([
            [s_y * shift_factor, -A * x0 / s_x],
            [0.0,                1.0 / s_x],
        ])
    else:  # y = A * exp(-x / tau) + C
        C = s_y * p_norm[2]
        tau = -1.0 / B if abs(B) > 1e-15 else np.inf
        popt = np.array([A, tau, C])
        d_tau_db = s_x / (b_n ** 2) if abs(b_n) > 1e-15 else np.nan
        T = np.array([
            [s_y * shift_factor, -A * x0 / s_x, 0.0],
            [0.0,                d_tau_db,      0.0],
            [0.0,                0.0,           s_y],
        ])

    pcov = T @ pcov_norm @ T.T
    perr = np.sqrt(np.maximum(np.diag(pcov), 0.0)) if np.all(np.isfinite(pcov)) else np.full(len(popt), np.nan)

    # Evaluate y_fit in normalized space to avoid any float64 overflow when x0 >> s_x
    y_pred = s_y * (a_n * np.exp(np.clip(b_n * u, -500, 500)) + (p_norm[2] if has_c else 0.0))
    residuals = y - y_pred

    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    red_chi2 = (
        float(np.sum((residuals / sigma) ** 2) / dof)
        if (sigma is not None and dof > 0)
        else np.nan
    )

    return {
        "x": x,
        "y": y,
        "y_err": sigma,
        "y_fit": y_pred,
        "residuals": residuals,
        "param_names": param_names,
        "popt": popt,
        "perr": perr,
        "r2": r2,
        "rmse": rmse,
        "red_chi2": red_chi2,
        "n_points": len(x),
        "func": func,
    }


def sanitize_filename(name):
    """Make a sheet name safe for filenames."""
    return re.sub(r'[\\/*?:"<>|]', "_", str(name)).strip() or "unnamed_sheet"


def make_excel_sheet_name(name, existing_names):
    """Ensure a sheet name obeys Excel's 31-character limit and uniqueness."""
    clean = re.sub(r'[\[\]:*?/\\]', "_", str(name)).strip()[:31] or "Sheet"
    candidate = clean
    idx = 1
    while candidate in existing_names:
        suffix = f"_{idx}"
        candidate = clean[: 31 - len(suffix)] + suffix
        idx += 1
    existing_names.add(candidate)
    return candidate


# =============================================================================
# Plotting
# =============================================================================

def build_fit_figure(
    fit_res,
    sheet_name,
    model_name,
    title_template,
    xlabel,
    ylabel,
    use_full_latex=False,
    show_params_box=True,
    log_y=False,
):
    """Create a Matplotlib Figure with LaTeX support for the fitted data."""
    matplotlib.rcParams["text.usetex"] = bool(use_full_latex)
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["mathtext.fontset"] = "cm"

    fig = Figure(figsize=(7.2, 5.0), dpi=150)
    ax = fig.add_subplot(111)

    x = fit_res["x"]
    y = fit_res["y"]
    y_err = fit_res["y_err"]
    func = fit_res["func"]
    popt = fit_res["popt"]
    perr = fit_res["perr"]

    if y_err is not None:
        ax.errorbar(
            x, y, yerr=y_err, fmt="o", markersize=4.5, capsize=2.5,
            color="#1f77b4", ecolor="#5b9bd5", alpha=0.85, label="Data"
        )
    else:
        ax.scatter(x, y, s=24, color="#1f77b4", alpha=0.85, label="Data")

    x_smooth = np.linspace(np.min(x), np.max(x), 500)
    y_smooth = func(x_smooth, *popt)
    ax.plot(x_smooth, y_smooth, color="#d62728", linewidth=2.0, label="Exponential Fit")

    if log_y:
        ax.set_yscale("log")

    # Allow {sheet} placeholder in title; escape underscores if full usetex is on
    display_sheet = sheet_name.replace("_", r"\_") if use_full_latex else sheet_name
    rendered_title = title_template.replace("{sheet}", display_sheet)

    ax.set_title(rendered_title, fontsize=13, pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=True, fontsize=9.5)

    if show_params_box:
        lines = [MODELS[model_name]["latex_eq"]]
        for pname, val, err in zip(fit_res["param_names"], popt, perr):
            p_tex = r"\tau" if pname == "tau" else pname
            if np.isfinite(err):
                lines.append(rf"${p_tex} = {val:.4g} \pm {err:.2g}$")
            else:
                lines.append(rf"${p_tex} = {val:.4g}$")
        lines.append(rf"$R^2 = {fit_res['r2']:.5f}$")
        if np.isfinite(fit_res["red_chi2"]):
            lines.append(rf"$\chi^2_{{\nu}} = {fit_res['red_chi2']:.3f}$")

        ax.text(
            0.03,
            0.97,
            "\n".join(lines),
            transform=ax.transAxes,
            fontsize=9.5,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.9),
        )

    fig.tight_layout()
    return fig


# =============================================================================
# Interactive GUI
# =============================================================================

class BatchExpFitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Batch Exponential Fitter (Excel + LaTeX Plots)")
        self.root.geometry("920x740")

        self.sheet_names = []
        self.sheet_columns = {}

        self._build_ui()

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        # --- 1. File & Output Paths ---
        file_frame = ttk.LabelFrame(main, text="1. Input & Output Paths", padding=8)
        file_frame.pack(fill=tk.X, pady=(0, 8))
        file_frame.columnconfigure(1, weight=1)

        self.input_path_var = tk.StringVar()
        self.output_excel_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()

        ttk.Label(file_frame, text="Input Excel File:").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=2)
        ttk.Entry(file_frame, textvariable=self.input_path_var).grid(row=0, column=1, sticky="ew", pady=2)
        ttk.Button(file_frame, text="Browse...", command=self.browse_input).grid(row=0, column=2, padx=(6, 0), pady=2)

        ttk.Label(file_frame, text="Output Excel File:").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=2)
        ttk.Entry(file_frame, textvariable=self.output_excel_var).grid(row=1, column=1, sticky="ew", pady=2)
        ttk.Button(file_frame, text="Browse...", command=self.browse_output_excel).grid(row=1, column=2, padx=(6, 0), pady=2)

        ttk.Label(file_frame, text="Output Plots Folder:").grid(row=2, column=0, sticky="w", padx=(0, 6), pady=2)
        ttk.Entry(file_frame, textvariable=self.output_dir_var).grid(row=2, column=1, sticky="ew", pady=2)
        ttk.Button(file_frame, text="Browse...", command=self.browse_output_dir).grid(row=2, column=2, padx=(6, 0), pady=2)

        # --- Middle Split: Sheets on left, Fit & Plot Settings on right ---
        mid = ttk.Frame(main)
        mid.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        mid.columnconfigure(0, weight=1)
        mid.columnconfigure(1, weight=2)
        mid.rowconfigure(0, weight=1)

        # --- 2. Sheet Selection ---
        sheet_frame = ttk.LabelFrame(mid, text="2. Sheets to Process", padding=8)
        sheet_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        sheet_frame.rowconfigure(0, weight=1)
        sheet_frame.columnconfigure(0, weight=1)

        self.sheet_listbox = tk.Listbox(sheet_frame, selectmode=tk.EXTENDED, exportselection=False)
        self.sheet_listbox.grid(row=0, column=0, sticky="nsew")
        self.sheet_listbox.bind("<<ListboxSelect>>", self.on_sheet_focus)

        sb = ttk.Scrollbar(sheet_frame, orient=tk.VERTICAL, command=self.sheet_listbox.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.sheet_listbox.config(yscrollcommand=sb.set)

        btn_row = ttk.Frame(sheet_frame)
        btn_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Button(btn_row, text="Select All", command=self.select_all_sheets).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(btn_row, text="Clear", command=self.clear_sheet_selection).pack(side=tk.LEFT)

        # --- 3. Columns, Model & LaTeX Plot Settings ---
        right_panel = ttk.Frame(mid)
        right_panel.grid(row=0, column=1, sticky="nsew")
        right_panel.columnconfigure(0, weight=1)

        # Columns & Model
        fit_frame = ttk.LabelFrame(right_panel, text="3. Columns & Fit Model", padding=8)
        fit_frame.pack(fill=tk.X, pady=(0, 8))
        fit_frame.columnconfigure(1, weight=1)
        fit_frame.columnconfigure(3, weight=1)

        self.x_col_var = tk.StringVar()
        self.y_col_var = tk.StringVar()
        self.yerr_col_var = tk.StringVar(value="(None)")
        self.model_var = tk.StringVar(value="y = A * exp(B * x) + C")
        self.xmin_var = tk.StringVar()
        self.xmax_var = tk.StringVar()

        ttk.Label(fit_frame, text="X Column:").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=3)
        self.x_col_cb = ttk.Combobox(fit_frame, textvariable=self.x_col_var)
        self.x_col_cb.grid(row=0, column=1, sticky="ew", padx=(0, 10), pady=3)

        ttk.Label(fit_frame, text="Y Column:").grid(row=0, column=2, sticky="w", padx=(0, 6), pady=3)
        self.y_col_cb = ttk.Combobox(fit_frame, textvariable=self.y_col_var)
        self.y_col_cb.grid(row=0, column=3, sticky="ew", pady=3)

        ttk.Label(fit_frame, text="Y Error (opt):").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=3)
        self.yerr_col_cb = ttk.Combobox(fit_frame, textvariable=self.yerr_col_var)
        self.yerr_col_cb.grid(row=1, column=1, sticky="ew", padx=(0, 10), pady=3)

        ttk.Label(fit_frame, text="Model:").grid(row=1, column=2, sticky="w", padx=(0, 6), pady=3)
        ttk.Combobox(
            fit_frame, textvariable=self.model_var, values=list(MODELS.keys()), state="readonly"
        ).grid(row=1, column=3, sticky="ew", pady=3)

        ttk.Label(fit_frame, text="X Min (opt):").grid(row=2, column=0, sticky="w", padx=(0, 6), pady=3)
        ttk.Entry(fit_frame, textvariable=self.xmin_var).grid(row=2, column=1, sticky="ew", padx=(0, 10), pady=3)

        ttk.Label(fit_frame, text="X Max (opt):").grid(row=2, column=2, sticky="w", padx=(0, 6), pady=3)
        ttk.Entry(fit_frame, textvariable=self.xmax_var).grid(row=2, column=3, sticky="ew", pady=3)

        # Plot & LaTeX Settings
        plot_frame = ttk.LabelFrame(right_panel, text="4. Plot Labels (LaTeX Supported via $...$)", padding=8)
        plot_frame.pack(fill=tk.X, pady=(0, 8))
        plot_frame.columnconfigure(1, weight=1)

        self.title_var = tk.StringVar(value=r"Exponential Fit: {sheet}")
        self.xlabel_var = tk.StringVar(value=r"$1/B\ (\mathrm{T}^{-1})$")
        self.ylabel_var = tk.StringVar(value=r"$\Delta R_{xx}\ (\Omega)$")
        self.format_var = tk.StringVar(value="png")
        self.dpi_var = tk.StringVar(value="300")

        self.use_full_latex_var = tk.BooleanVar(value=False)
        self.show_params_var = tk.BooleanVar(value=True)
        self.log_y_var = tk.BooleanVar(value=False)
        self.include_sheet_tabs_var = tk.BooleanVar(value=True)

        ttk.Label(plot_frame, text="Title ({sheet} = name):").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=3)
        ttk.Entry(plot_frame, textvariable=self.title_var).grid(row=0, column=1, columnspan=3, sticky="ew", pady=3)

        ttk.Label(plot_frame, text="X-Axis Label:").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=3)
        ttk.Entry(plot_frame, textvariable=self.xlabel_var).grid(row=1, column=1, columnspan=3, sticky="ew", pady=3)

        ttk.Label(plot_frame, text="Y-Axis Label:").grid(row=2, column=0, sticky="w", padx=(0, 6), pady=3)
        ttk.Entry(plot_frame, textvariable=self.ylabel_var).grid(row=2, column=1, columnspan=3, sticky="ew", pady=3)

        ttk.Label(plot_frame, text="Image Format:").grid(row=3, column=0, sticky="w", padx=(0, 6), pady=3)
        ttk.Combobox(
            plot_frame, textvariable=self.format_var, values=["png", "pdf", "svg", "jpg"], state="readonly", width=10
        ).grid(row=3, column=1, sticky="w", pady=3)

        ttk.Label(plot_frame, text="DPI:").grid(row=3, column=2, sticky="e", padx=(0, 6), pady=3)
        ttk.Entry(plot_frame, textvariable=self.dpi_var, width=8).grid(row=3, column=3, sticky="w", pady=3)

        opts = ttk.Frame(plot_frame)
        opts.grid(row=4, column=0, columnspan=4, sticky="w", pady=(6, 0))
        ttk.Checkbutton(opts, text="Show fit box on plot", variable=self.show_params_var).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Checkbutton(opts, text="Log Y-axis", variable=self.log_y_var).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Checkbutton(opts, text="Full LaTeX (usetex)", variable=self.use_full_latex_var).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Checkbutton(opts, text="Include per-sheet fit data tabs", variable=self.include_sheet_tabs_var).pack(side=tk.LEFT)

        # --- Action Buttons & Progress ---
        action_frame = ttk.Frame(main)
        action_frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Button(action_frame, text="Preview Selected Sheet", command=self.preview_selected).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(action_frame, text="Run Batch Fit", command=self.run_batch).pack(side=tk.LEFT)

        self.progress = ttk.Progressbar(action_frame, mode="determinate")
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(12, 0))

        # --- Log Output ---
        log_frame = ttk.LabelFrame(main, text="Log", padding=6)
        log_frame.pack(fill=tk.BOTH, expand=False)
        self.log_text = tk.Text(log_frame, height=8, state=tk.DISABLED, font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def log(self, msg):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.root.update_idletasks()

    def browse_input(self):
        path = filedialog.askopenfilename(
            title="Select Input Excel File",
            filetypes=[("Excel files", "*.xlsx *.xls *.xlsm"), ("All files", "*.*")],
        )
        if not path:
            return
        self.input_path_var.set(path)

        base_dir = os.path.dirname(path)
        stem = os.path.splitext(os.path.basename(path))[0]
        self.output_excel_var.set(os.path.join(base_dir, f"{stem}_exp_fits.xlsx"))
        self.output_dir_var.set(os.path.join(base_dir, f"{stem}_fit_plots"))

        self.load_workbook_metadata(path)

    def browse_output_excel(self):
        path = filedialog.asksaveasfilename(
            title="Save Results Excel File",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
        )
        if path:
            self.output_excel_var.set(path)

    def browse_output_dir(self):
        path = filedialog.askdirectory(title="Select Output Folder for Plot Images")
        if path:
            self.output_dir_var.set(path)

    def load_workbook_metadata(self, path):
        try:
            xl = pd.ExcelFile(path)
            self.sheet_names = xl.sheet_names
            self.sheet_columns.clear()

            self.sheet_listbox.delete(0, tk.END)
            for s in self.sheet_names:
                self.sheet_listbox.insert(tk.END, s)
                df_head = xl.parse(s, nrows=2)
                self.sheet_columns[s] = [str(c) for c in df_head.columns]

            self.select_all_sheets()
            if self.sheet_names:
                self._populate_column_boxes(self.sheet_columns[self.sheet_names[0]])

            self.log(f"Loaded '{os.path.basename(path)}' with {len(self.sheet_names)} sheet(s).")
        except Exception as e:
            messagebox.showerror("Error Loading File", str(e))

    def _populate_column_boxes(self, cols):
        self.x_col_cb["values"] = cols
        self.y_col_cb["values"] = cols
        self.yerr_col_cb["values"] = ["(None)"] + cols

        if cols:
            if self.x_col_var.get() not in cols:
                self.x_col_var.set(cols[0])
            if self.y_col_var.get() not in cols:
                self.y_col_var.set(cols[1] if len(cols) > 1 else cols[0])
            if self.yerr_col_var.get() not in ["(None)"] + cols:
                self.yerr_col_var.set("(None)")

    def on_sheet_focus(self, _event):
        sel = self.sheet_listbox.curselection()
        if len(sel) == 1:
            sname = self.sheet_names[sel[0]]
            if sname in self.sheet_columns:
                self._populate_column_boxes(self.sheet_columns[sname])

    def select_all_sheets(self):
        self.sheet_listbox.select_set(0, tk.END)

    def clear_sheet_selection(self):
        self.sheet_listbox.selection_clear(0, tk.END)

    def _extract_xy(self, df):
        x_col = self.x_col_var.get().strip()
        y_col = self.y_col_var.get().strip()
        yerr_col = self.yerr_col_var.get().strip()

        df.columns = [str(c) for c in df.columns]
        if x_col not in df.columns or y_col not in df.columns:
            raise KeyError(f"Missing column(s): '{x_col}' or '{y_col}'.")

        x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)

        y_err = None
        if yerr_col and yerr_col != "(None)":
            if yerr_col not in df.columns:
                raise KeyError(f"Missing Y-error column: '{yerr_col}'.")
            y_err = pd.to_numeric(df[yerr_col], errors="coerce").to_numpy(dtype=float)
            mask &= np.isfinite(y_err) & (y_err > 0)

        xmin_str = self.xmin_var.get().strip()
        xmax_str = self.xmax_var.get().strip()
        if xmin_str:
            mask &= x >= float(xmin_str)
        if xmax_str:
            mask &= x <= float(xmax_str)

        return x[mask], y[mask], (y_err[mask] if y_err is not None else None)

    # -------------------------------------------------------------------------
    # Preview & Batch Execution
    # -------------------------------------------------------------------------

    def preview_selected(self):
        path = self.input_path_var.get().strip()
        sel = self.sheet_listbox.curselection()
        if not path or not os.path.isfile(path):
            messagebox.showwarning("No File", "Please select a valid input Excel file first.")
            return
        if not sel:
            messagebox.showwarning("No Sheet", "Please select at least one sheet to preview.")
            return

        sheet_name = self.sheet_names[sel[0]]
        try:
            df = pd.read_excel(path, sheet_name=sheet_name)
            x, y, y_err = self._extract_xy(df)
            fit_res = perform_exponential_fit(x, y, y_err, self.model_var.get())
            fig = build_fit_figure(
                fit_res,
                sheet_name=sheet_name,
                model_name=self.model_var.get(),
                title_template=self.title_var.get(),
                xlabel=self.xlabel_var.get(),
                ylabel=self.ylabel_var.get(),
                use_full_latex=self.use_full_latex_var.get(),
                show_params_box=self.show_params_var.get(),
                log_y=self.log_y_var.get(),
            )

            win = tk.Toplevel(self.root)
            win.title(f"Preview: {sheet_name}")
            canvas = FigureCanvasTkAgg(fig, master=win)
            canvas.draw()
            toolbar = NavigationToolbar2Tk(canvas, win)
            toolbar.update()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        except Exception as e:
            messagebox.showerror("Preview Error", f"Sheet '{sheet_name}':\n{e}")

    def run_batch(self):
        in_path = self.input_path_var.get().strip()
        out_excel = self.output_excel_var.get().strip()
        out_dir = self.output_dir_var.get().strip()
        sel = self.sheet_listbox.curselection()

        if not in_path or not os.path.isfile(in_path):
            messagebox.showwarning("Missing Input", "Please select a valid input Excel file.")
            return
        if not sel:
            messagebox.showwarning("No Sheets", "Please select at least one sheet to process.")
            return
        if not out_excel or not out_dir:
            messagebox.showwarning("Missing Output", "Please specify both the output Excel file and plots folder.")
            return

        try:
            dpi = int(self.dpi_var.get().strip())
        except ValueError:
            messagebox.showwarning("Invalid DPI", "DPI must be an integer (e.g., 300).")
            return

        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(out_excel)), exist_ok=True)

        selected_sheets = [self.sheet_names[i] for i in sel]
        model_name = self.model_var.get()
        img_fmt = self.format_var.get()

        self.progress["maximum"] = len(selected_sheets)
        self.progress["value"] = 0

        summary_rows = []
        per_sheet_dfs = {}
        used_tab_names = {"Fit_Summary"}

        self.log(f"--- Starting batch fit ({len(selected_sheets)} sheets, model: {model_name}) ---")

        xl = pd.ExcelFile(in_path)

        for idx, sheet_name in enumerate(selected_sheets, start=1):
            row = {"Sheet": sheet_name, "Model": model_name, "Status": "OK"}
            try:
                df = xl.parse(sheet_name)
                x, y, y_err = self._extract_xy(df)
                fit_res = perform_exponential_fit(x, y, y_err, model_name)

                for pname, val, err in zip(fit_res["param_names"], fit_res["popt"], fit_res["perr"]):
                    row[pname] = val
                    row[f"{pname}_err"] = err

                row["R_squared"] = fit_res["r2"]
                row["RMSE"] = fit_res["rmse"]
                if np.isfinite(fit_res["red_chi2"]):
                    row["Reduced_Chi2"] = fit_res["red_chi2"]
                row["N_points"] = fit_res["n_points"]

                # Save plot named after sheet
                fig = build_fit_figure(
                    fit_res,
                    sheet_name=sheet_name,
                    model_name=model_name,
                    title_template=self.title_var.get(),
                    xlabel=self.xlabel_var.get(),
                    ylabel=self.ylabel_var.get(),
                    use_full_latex=self.use_full_latex_var.get(),
                    show_params_box=self.show_params_var.get(),
                    log_y=self.log_y_var.get(),
                )
                img_name = f"{sanitize_filename(sheet_name)}.{img_fmt}"
                img_path = os.path.join(out_dir, img_name)
                fig.savefig(img_path, dpi=dpi, bbox_inches="tight")
                row["Plot_File"] = img_name

                if self.include_sheet_tabs_var.get():
                    fit_df = pd.DataFrame({
                        self.x_col_var.get(): fit_res["x"],
                        self.y_col_var.get(): fit_res["y"],
                        "y_fit": fit_res["y_fit"],
                        "residual": fit_res["residuals"],
                    })
                    if fit_res["y_err"] is not None:
                        fit_df.insert(2, self.yerr_col_var.get(), fit_res["y_err"])
                    tab_name = make_excel_sheet_name(sheet_name, used_tab_names)
                    per_sheet_dfs[tab_name] = fit_df

                self.log(f"[OK] {sheet_name} -> R^2 = {fit_res['r2']:.5f}, saved {img_name}")

            except Exception as e:
                row["Status"] = f"ERROR: {e}"
                self.log(f"[ERR] {sheet_name}: {e}")
                traceback.print_exc()

            summary_rows.append(row)
            self.progress["value"] = idx
            self.root.update()

        # Write output Excel workbook
        try:
            summary_df = pd.DataFrame(summary_rows)
            with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
                summary_df.to_excel(writer, sheet_name="Fit_Summary", index=False)
                for tab_name, fdf in per_sheet_dfs.items():
                    fdf.to_excel(writer, sheet_name=tab_name, index=False)

            self.log(f"--- Done! Results saved to '{out_excel}' and plots to '{out_dir}' ---")
            messagebox.showinfo(
                "Batch Complete",
                f"Processed {len(selected_sheets)} sheet(s).\n\n"
                f"Excel: {out_excel}\n"
                f"Plots: {out_dir}",
            )
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to write output Excel file:\n{e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = BatchExpFitApp(root)
    root.mainloop()