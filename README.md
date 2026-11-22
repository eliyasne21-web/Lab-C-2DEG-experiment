This page is a repository for a 2DEG experiment analysis by and Idan Sosnovsky Eliya Sne.

The experiment is aimed at characterizing the transport properties of a 2D electron gas and the various phenomena that arise in it as a result of energy level quantization, which typically appear in low-single-digit kelvin cryogenic temperatures. the experiments are performed on a GaAs/AlGaAs heterostructure sample in a four-terminal Hall bar arrangement.

Contained within the repository are the following:
1) Under `raw data` you can find the original experimental mesurements the analysis is based on.
2) `main.tex` - this is the actual LaTeX written analysis.
3) `2DEG_IdanSosnovsky_EliyaSne.pdf` - the analysis in PDF format.
4) Under the `analysis code` folder you'lle find excel sheets and python scripts that were used in the analysis. more details below.
5) Various figures produced during the analysis, including but not limited to those seen in the final paper.

Analysis code and sheets:
* `2DEG.xlsx` - contains the raw data from the full temperature and field sweeps in an aproachable format.
* `gateSweeps` - contains the raw data from the variable gate mesurements.
* `SdH_amplitudes.xlsx` - contains identified extrema in the R_xx mesurments and perliminary analysis of the resulting frequency.
* `gateSweeps_exp_fits.xlsx` - contains exponential fits for the gate sweep data (not in the final paper). Said fits were aquired using the `exponentialFits.py` script and their plots can be found at `gateSweep_plots`.
* `flatZoneFinder.py` - is a script used for the identification of minima, maxima, and plateaus in any of the various measurments. Its it operated using the `flatFind.py` script, which functions as a freindlier user interface.


