2DEG Lab C / PRB Overleaf figures
=================================
1. Open your Overleaf project.
2. Create a folder named figures.
3. Upload all 9 PDF figures from this folder into the Overleaf figures folder.
4. Ensure main.tex includes \usepackage{graphicx}.
5. Open figures_for_report.tex and paste each figure environment into the Results subsection indicated by its preceding comment.
6. Use \ref{fig:...} for references; figures will float to appropriate positions in REVTeX.
7. If you want a figure spanning both columns, use figure* instead of figure, and \textwidth instead of \columnwidth.

Suggested narrative insertion examples:
'Figure~\ref{fig:cooldown} shows the resistance during cooldown.'
'Figure~\ref{fig:hall_fits} shows the low-field Hall fits used to extract the carrier densities in Table~\ref{tab:hall}.'
'Figure~\ref{fig:gate_density} summarizes the inferred gate dependence of carrier density.'
'Figure~\ref{fig:magnetotransport} shows magnetotransport structure in the raw lock-in channels.'
'Figure~\ref{fig:temperature_drift} documents field-sweep temperature drift.'

Cautions:
- All plots are generated from the provided text files; there is no invented experimental data.
- Assigning SR510 No. 1 to Rxx and No. 2 to Rxy is provisional.
- Do not label the large oscillations as established SdH or the Hall curves as quantitatively quantized without further checks.
- The lower temperature field-sweep thermometer reads 0 after the initial field range: these invalid values are excluded.
- Some files contain repeated 25565-ohm readings, possible range limit/saturation.
- Figures are available as vector PDF and raster PNG. Use PDF for Overleaf.
- build_2deg_figures.py at the top level of the delivered bundle recreates all figures from the raw text files.

Fit results over 0.01--0.50 tesla (Hall channel):
Vg -0.5 V: slope 2037.6082 ohm/T, n=3.06315e11 cm^-2
Vg  0.0 V: slope 1231.8737 ohm/T, n=5.06668e11 cm^-2
Vg +0.5 V: slope  849.6282 ohm/T, n=7.34617e11 cm^-2
