import subprocess
import sys
import shlex

def main():
    print("========================================")
    print("      2DEG Flat Zone Finder Setup       ")
    print("========================================")

    in_file = input("1. Enter input Excel file path: ").strip()
    in_sheet = input("2. Enter input sheet name: ").strip()
    x_col = input("3. Enter X-axis column (name or number): ").strip()
    y_col = input("4. Enter Y-axis column (name or number): ").strip()

    print("\n--- Output Settings ---")
    out_file = input("5. Enter output Excel file path: ").strip()
    out_sheet = input("6. Enter output sheet name: ").strip()

    print("\n--- Mode & Analysis Parameters ---")
    kind = input("7. Search mode? (all / max / min) [default: all]: ").strip().lower()
    if kind not in ['max', 'min']:
        kind = "all"

    command = [
        sys.executable, "flatZoneFinder.py",
        "-i", in_file,
        "--in_sheet", in_sheet,
        "-x", x_col,
        "-y", y_col,
        "-o", out_file,
        "--out_sheet", out_sheet,
        "-k", kind
    ]

    if kind == 'all':
        threshold = input("8. Enter derivative threshold (e.g., 0.05): ").strip()
        min_points = input("9. Enter minimum consecutive points (default 5): ").strip() or "5"
        command.extend(["-t", threshold, "-m", min_points])
    else:
        prominence = input("8. Enter minimum peak prominence (leave blank for none): ").strip()
        distance = input("9. Enter minimum point distance between peaks (default 5): ").strip() or "5"
        command.extend(["-m", distance])
        if prominence:
            command.extend(["--prominence", prominence])

    print("\n--- Smoothing Settings ---")
    window = input("10. Enter filter window length (odd number, default 15): ").strip() or "15"
    poly = input("11. Enter polynomial order (default 2): ").strip() or "2"
    command.extend(["-w", window, "-p", poly])

    command_str = shlex.join(command)

    print("\n========================================")
    print("Generated Command Line (Copy below):")
    print("----------------------------------------")
    print(command_str)
    print("========================================\n")

    print("Executing flatZoneFinder...\n")

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nExecution failed with error code {e.returncode}.")
    except FileNotFoundError:
        print("\nError: Could not find 'flatZoneFinder.py'. Please ensure it is in the same folder as this script.")

if __name__ == "__main__":
    main()