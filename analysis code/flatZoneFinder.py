import argparse
import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks

def _sanitize_savgol_params(n_points, window, poly, min_poly=2):
    poly = max(poly, min_poly)
    if n_points < window:
        window = n_points if n_points % 2 != 0 else n_points - 1
    if window % 2 == 0:
        window += 1
    if poly >= window:
        poly = window - 1
    return window, poly

def find_flat_sections(x, y, derivative_threshold, min_points, window, poly):
    """
    Finds flat sections (e.g., QHE plateaus) where |dy/dx| < threshold.
    """
    window, poly = _sanitize_savgol_params(len(x), window, poly, min_poly=1)

    dx = np.median(np.abs(np.diff(x)))
    if dx == 0:
        dx = 1.0

    dydx = savgol_filter(y, window_length=window, polyorder=poly, deriv=1, delta=dx)
    flat_indices = np.where(np.abs(dydx) < derivative_threshold)[0]

    if len(flat_indices) == 0:
        return pd.DataFrame()

    split_points = np.where(np.diff(flat_indices) != 1)[0] + 1
    sections = np.split(flat_indices, split_points)

    results = []
    for sec in sections:
        n = len(sec)
        if n >= min_points:
            x_sec = x[sec]
            y_sec = y[sec]

            results.append({
                'Feature': 'Flat',
                'Mean_X': np.mean(x_sec),
                'Error_X': np.std(x_sec, ddof=1) / np.sqrt(n) if n > 1 else 0.0,
                'Mean_Y': np.mean(y_sec),
                'Error_Y': np.std(y_sec, ddof=1) / np.sqrt(n) if n > 1 else 0.0,
                'N_points': n,
                'Start_Index': sec[0],
                'End_Index': sec[-1]
            })

    return pd.DataFrame(results)

def find_extrema(x, y, window, poly, kind, prominence=None, distance=1):
    """
    Finds local maxima or minima using scipy.signal.find_peaks on smoothed data,
    with errors scaled by the local second derivative d2y/dx2.
    """
    # polyorder must be at least 2 to compute a non-zero 2nd derivative
    window, poly = _sanitize_savgol_params(len(x), window, poly, min_poly=2)

    dx = np.median(np.abs(np.diff(x)))
    if dx == 0:
        dx = 1.0

    y_smooth = savgol_filter(y, window_length=window, polyorder=poly, deriv=0)
    d2ydx2 = savgol_filter(y, window_length=window, polyorder=poly, deriv=2, delta=dx)

    # Estimate baseline slope noise from the smoothing residuals
    sigma_y = np.std(y - y_smooth, ddof=1)
    sigma_dydx = sigma_y / dx

    target = y_smooth if kind == 'max' else -y_smooth
    peaks, _ = find_peaks(target, prominence=prominence, distance=max(1, distance))

    if len(peaks) == 0:
        return pd.DataFrame()

    results = []
    label = 'Max' if kind == 'max' else 'Min'

    for idx in peaks:
        d2 = d2ydx2[idx]
        abs_d2 = np.abs(d2)

        # X error ~ 1 / |y''|, Y error ~ |y''|
        err_x = (sigma_dydx / abs_d2) if abs_d2 > 0 else np.nan
        err_y = 0.5 * abs_d2 * (dx ** 2)

        results.append({
            'Feature': label,
            'X': x[idx],
            'Error_X': err_x,
            'Y': y_smooth[idx],
            'Error_Y': err_y,
            'd2y_dx2': d2,
            'Index': idx
        })

    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description="Find flat sections or extrema in 2DEG Excel data.")

    parser.add_argument('-i', '--input', required=True, help="Path to the input Excel file.")
    parser.add_argument('--in_sheet', required=True, help="Name of the input sheet.")
    parser.add_argument('-x', '--xcol', required=True, help="Name or index of the X data column.")
    parser.add_argument('-y', '--ycol', required=True, help="Name or index of the Y data column.")

    parser.add_argument('-o', '--output', required=True, help="Path to the output Excel file.")
    parser.add_argument('--out_sheet', required=True, help="Name of the output sheet.")

    parser.add_argument('-k', '--kind', choices=['all', 'max', 'min'], default='all',
                        help="'all' for flat zones (QHE), 'max'/'min' for scipy extremum finding (SdH).")
    parser.add_argument('-t', '--threshold', type=float, default=None,
                        help="Max |dy/dx| threshold (required when --kind is 'all').")
    parser.add_argument('-m', '--min_points', type=int, default=5,
                        help="Min consecutive points for flat zones, or min point distance between extrema.")
    parser.add_argument('--prominence', type=float, default=None,
                        help="Minimum peak prominence for 'max'/'min' modes (optional).")

    parser.add_argument('-w', '--window', type=int, default=15, help="Savitzky-Golay window length (odd).")
    parser.add_argument('-p', '--poly', type=int, default=2, help="Savitzky-Golay polynomial order.")

    args = parser.parse_args()

    if args.kind == 'all' and args.threshold is None:
        parser.error("--threshold (-t) is required when --kind is 'all'.")

    print(f"\n[flatZoneFinder] Loading '{args.in_sheet}' from {args.input}...")
    try:
        df_in = pd.read_excel(args.input, sheet_name=args.in_sheet)
    except Exception as e:
        print(f"[flatZoneFinder] Error reading input file/sheet: {e}")
        return

    df_in.columns = [str(col).strip() for col in df_in.columns]

    def extract_col(df, col_identifier):
        col_identifier = col_identifier.strip()
        if col_identifier in df.columns:
            return df[col_identifier].values
        if col_identifier.isdigit():
            idx = int(col_identifier)
            if 0 <= idx < len(df.columns):
                return df.iloc[:, idx].values
        return None

    x_raw = extract_col(df_in, args.xcol)
    y_raw = extract_col(df_in, args.ycol)

    if x_raw is None or y_raw is None:
        print(f"[flatZoneFinder] Error: Could not find columns '{args.xcol}' and/or '{args.ycol}'.")
        return

    x_num = pd.to_numeric(x_raw, errors='coerce')
    y_num = pd.to_numeric(y_raw, errors='coerce')

    valid_mask = ~np.isnan(x_num) & ~np.isnan(y_num)
    x_data = x_num[valid_mask]
    y_data = y_num[valid_mask]

    if len(x_data) == 0:
        print("[flatZoneFinder] Error: No valid numeric data found in the selected columns.")
        return

    if args.kind == 'all':
        print("[flatZoneFinder] Searching for flat sections...")
        df_results = find_flat_sections(
            x_data, y_data, args.threshold, args.min_points, args.window, args.poly
        )
    else:
        print(f"[flatZoneFinder] Searching for '{args.kind}' extrema using scipy.signal.find_peaks...")
        df_results = find_extrema(
            x_data, y_data, args.window, args.poly, args.kind,
            prominence=args.prominence, distance=args.min_points
        )

    if df_results.empty:
        print("[flatZoneFinder] No features found matching your criteria.")
        return

    print(f"[flatZoneFinder] Found {len(df_results)} features. Saving to {args.output} (Sheet: {args.out_sheet})...")

    try:
        if os.path.exists(args.output):
            with pd.ExcelWriter(args.output, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                df_results.to_excel(writer, sheet_name=args.out_sheet, index=False)
        else:
            with pd.ExcelWriter(args.output, engine='openpyxl') as writer:
                df_results.to_excel(writer, sheet_name=args.out_sheet, index=False)
        print("[flatZoneFinder] Done!")
    except Exception as e:
        print(f"[flatZoneFinder] Error writing to output file: {e}")

if __name__ == "__main__":
    main()