#!/usr/bin/env python
"""Stellar parameterization using flux ratios from optical spectra.

This is the main routine of ATHOS responsible for reading the necessary 
parameters from 'parameters.py' and loading the spectrum information from the 
file 'input_specs'. Subsequently, depending on the settings, the workflow is 
executed either in multi- or single-thread mode. Finally, the output file 
'output_file' is generated. For a detailed documentaion, see 'athos_utils' or 
the paper [1].

References
----------
.. [1] Astronomy & Astrophysics, "ATHOS: On-the-fly stellar parameter 
determination of FGK stars based on flux ratios from optical spectra", 
M. Hanke, C. J. Hansen, A. Koch and E. K. Grebel, 2018, A&A, 619, A134

"""
import numpy as np
import athos_utils

from parameters import *
from multiprocessing import cpu_count
from joblib import Parallel, delayed

# Check whether wave_keywd and/or flux_keywd is set, otherwise set it to 'None'
if 'wave_keywd' in locals():
    pass
else:
    wave_keywd = None
    flux_keywd = None

# Read input file
with open(input_specs, 'r') as f_in:
    input_list = [line.rstrip('\n') for line in f_in]

file_list, v_topo_list, p_weights, max_file_len = [], [], [], 0
for line in input_list:
    if line[0] != '#':
        line_spl = line.split()
        if len(line_spl[0]) > max_file_len:
            max_file_len = len(line_spl[0])
        file_list.append(line_spl[0])
        if len(line_spl) > 1:
            v_topo_list.append(float(line_spl[1]))
            if len(line_spl) > 2:
                p_weights.append(np.array(line_spl[2:]).astype(float))
            else:
                p_weights.append(None)
        else:
            p_weights.append(None)
            v_topo_list.append(None)

# Split input for multiprocessing
if n_threads == 1:
    results = []
    for i in range(len(file_list)):
        results.append(list(athos_utils.analyze_spectrum(file_list[i], dtype, wunit, p_weights = p_weights[i], tell_rejection = tell_rejection, v_topo = v_topo_list[i], y_err = None, R = R, wave_keywd = wave_keywd, flux_keywd = flux_keywd)))
        
else:
    if n_threads == -1:
        num_avail = cpu_count()
    else:
        num_avail = n_threads
        
    thread_lists = []
    int_div = int(len(file_list)/num_avail)
    for i in range(num_avail):
        if i == num_avail - 1:
            thread_lists.append([file_list[i*int_div:], p_weights[i*int_div:], v_topo_list[i*int_div:]])
        else:
            thread_lists.append([file_list[i*int_div:(i+1)*int_div], p_weights[i*int_div:(i+1)*int_div], v_topo_list[i*int_div:(i+1)*int_div]])
        
    res = Parallel(n_jobs=n_threads)(delayed(athos_utils.parallelization_wrapper)(thread_lists[i][0], thread_lists[i][1], thread_lists[i][2], dtype, wunit, tell_rejection, yerr=None, R=R, wave_keywd = wave_keywd, flux_keywd = flux_keywd) for i in range(num_avail))
    
    results = []
    for r in res:
        results += r
        
# Write output file        
lines_out = ['#spectrum Teff Teff_err_stat Teff_err_sys [Fe/H] [Fe/H]_err_stat [Fe/H]_err_sys logg logg_err_stat logg_err_sys\n']
for i, line in enumerate(results):
    lines_out.append('{:{file_len}s}  {:5.0f}  {:4.0f}  {:4.0f}  {:5.2f}  {:4.2f}  {:4.2f}  {:4.2f}  {:4.2f}  {:4.2f}\n'.format(file_list[i], *line[:9], file_len=max_file_len))
    
with open(output_file, 'w') as f_out:
    f_out.writelines(lines_out)

if 'verbose' in locals():
    if verbose:
        fmt_string = '{:{file_len}s}  ' + '{:5.0f}  ' * len(results[0][9]) + '{:5.2f}  ' * len(results[0][10]) + '{:4.2f}  ' * len(results[0][11]) + '\n' 
        lines_out = ['#spectrum, individual Teff, individual [Fe/H], individual logg\n']
        for i, line in enumerate(results):
            lines_out.append(fmt_string.format(file_list[i], *line[9], *line[10], *line[11], file_len=max_file_len))
            
        with open(output_file + '_verbose', 'w') as f_out:
            f_out.writelines(lines_out)
