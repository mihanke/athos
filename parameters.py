input_specs = 'path/to/input/file'  # A string pointing to the file with information about the input spectra
output_file = 'path/to/output/file' # A string specifying the desired output file 
dtype = 'lin'         # one of 'lin', 'log10', or 'ln'
wunit = 'aa'          # either 'aa' or 'nm'
R = 45000             # a number <= 45000
tell_rejection = True # Either True or False. If True, the range lambda - lamda_i/R < lambda < lambda + lambda_i/R will be masked for each internally stored telluric lambda_i
n_threads = -1        # number of threads for parallelization; all available cores/threads if set to -1
# wave_keywd = None   # Wavelength keyword for fits binary table spectra.
# flux_keywd = None   # Flux keyword for fits binary table spectra.
