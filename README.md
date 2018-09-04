# ATHOS
## On-the-fly stellar parameter determination of FGK stars based on flux ratios from optical spectra
*M. Hanke, C. J. Hansen, A. Koch, and E. K. Grebel*

LOGO

ATHOS (__A__ __T__ ool for __HO__ mogenizing __S__ tellar parameters) is __A__ (non-exhaustive, readers are encouraged to adapt the tool to their needs!) computational implementation of the spectroscopic stellar parameterization method outlined in __LINK TO PAPER__. Once configured properly, it will measure flux ratios in the input spectra and deduce the stellar parameters *effective temperature*, *iron abundance* (a.k.a [Fe/H]), and *surface gravity* by employing pre-defined analytical relations. The code is written in Python and has been tested to work properly with Python 2.7+ and Python 3.4+. ATHOS can be configured to run in parallel in an arbitrary number of threads, thus enabling the fast and efficient analysis of huge datasets. 

Requirements
---

* `python` 2.7+ or 3.4+
* `numpy` 1.12+
* `astropy` 2.0+
* `pandas` 0.22+
* `multiprocessing` 0.7+
* `joblib` 0.12+

Input data
---
The routines are designed to deal with *__one-dimensional, optical__* stellar spectra that are *__shifted to the stellar rest frame__*. ATHOS supports several types of file structures, among which are standard 1D fits spectra, fits binary tables, numpy arrays, and plain text (see function `athos_utils.read_spectrum` for details).

Usage
===
In order to be able to execute ATHOS, copy the files `athos.py` and `athos_utils.py`, as well as the folder `/coefficients` to the same local directory. After the modifying the parameter file `parameters.py` (see next section), the code can be run from konsole by executing

```bash
$ python athos.py
```
everytime ATHOS is used, or by making `athos.py` executable
```bash
$ chmod +x athos.py
```
once and running it using
```bash
$ ./athos.py
```
on every subsequent call.

The parameter file `parameters.py`
---
ATHOS is initialized via parameters read from a file called `parameters.py`, which should be located in the same directory as `athos.py`. An example follows below (included in this repository):

```python
input_specs = 'path/to/input/file'  # A string pointing to the file with information about the input spectra
output_file = 'path/to/output/file' # A string specifying the desired output file 
dtype = 'lin'         # one of 'lin', 'log10', or 'ln'
wunit = 'aa'          # either 'aa' or 'nm'
R = 45000             # a number <= 45000
tell_rejection = True # Either True or False. If True, the range lambda - lamda_i/R < lambda < lambda + lambda_i/R will be masked for each internally stored telluric lambda_i
n_threads = -1        # number of threads for parallelization; all available cores/threads if set to -1
```
The following seven parameters must be set:
* `input_specs`: A string containing the (absolute or relative) path to the input text file that stores the information about the spectra (see next section).
* `output_file`: A string telling ATHOS where to save the output results.
* `dtype`: A string denoting the dispersion type of the input spectra. Valid options are 'lin', 'log10', or 'ln'.
* `wunit`: The wavelength unit can either be Angstroms ('aa') or nanometers ('nm').
* `R`: The resolution of the spectrograph.
* `tell_rejection`: A flag specifying whether a telluric rejection should be performed. If `tell_rejection` is set to `True`, the relative velocity of the topocenter has to be provided in the file `input_specs` (see next section).
* `n_threads`: The number of threads used for parallel computation. A value of `-1` indicates that all available cores/threads should be used.

The input file `input_specs`
---
Each line in the text file `input_specs` should contain the information about one input spectrum. Columns must be separated by whitespace. The lines should obey the following structure:
* __1st column__: The absolute or relative path to the spectrum file.
* __2nd column__ (optional): The relative velocity of the topocenter, `v_topo`, in km/s. The sign convention is such that if the input spectrum is blue-shifted w.r.t. the topocentric rest frame `sign(v_topo) = +1`, and `sign(v_topo) = -1` otherwise. This parameter is only needed if `tell_rejection` is set to `True`.
