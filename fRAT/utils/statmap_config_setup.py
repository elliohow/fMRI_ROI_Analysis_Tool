Statistical_maps = {
    'General settings': {'type': 'subheading'},

    'verbose': {'type': 'CheckButton', 'Recommended': 'true', 'label': 'Verbose statistical map stages',
                'Description': 'true or false. Print progress to terminal.'},

    'multicore_processing': {'type': 'CheckButton', 'Recommended': 'true',
                             'Description': 'true or false. Use multicore processing during analysis? '
                                            'Multicore processing currently works within participants not between them.'
                                            ' Recommended: true'},

    'max_core_usage': {'type': 'OptionMenu', 'Recommended': 'max', 'Options': ['max', 6, 5, 4, 3, 2, 1],
                       'save_as': 'string',
                       'Description': "'max' to select number of cores available on the system, alternatively an int to"
                                      " manually select number of cores to use. Recommended: 'max'"},

    'base_folder': {'type': 'Entry', 'Recommended': "", 'save_as': 'string',
                    'label': 'Base folder location',
                    'Description': 'Either the absolute location of the folder containing the subjects or blank, '
                                   'if blank then a browser window will allow you to search for the files at '
                                   'runtime.'},

    'input_folder_name': {'type': 'Entry', 'Recommended': "func", 'save_as': 'string',
                          'Description': 'Folder found in each subjects directory containing the files to be analysed.'
                          },

    'output_folder_name': {'type': 'Entry', 'Recommended': 'DEFAULT', 'save_as': 'string',
                           'Description': 'Directory to save output. If set to DEFAULT, the default name for the '
                                          'statistical map created will be used. '
                                          'Recommended: DEFAULT'},

    'High pass filtering': {'type': 'subheading'},

    'temporal_filter': {'type': 'CheckButton', 'Recommended': 'true',
                        'Description': 'true or false. Use a high pass filter to remove low frequency drift. '
                                       'Recommended: true'},

    'highpass_filter_cutoff': {'type': 'Entry', 'Recommended': 0.01,
                               'label': 'Highpass filter cutoff frequency (Hz)',
                               'Description': 'Highpass filter cutoff frequency converted into sigma in seconds using '
                                              'the formula 1/(2*f*TR). Recommended: 0.01'},

    'Motion correction': {'type': 'subheading'},

    'remove_motion_outliers': {'type': 'CheckButton', 'Recommended': 'true',
                               'Description': 'true or false. Use fsl_motion_outliers to remove motion outliers'
                                              '(uses default fsl_motion_outliers settings). Recommended: true'},

    'motion_correction': {'type': 'CheckButton', 'Recommended': 'true',
                          'Description': 'true or false. Use MCFLIRT to motion correct volumes '
                                         '(uses default MCFLIRT settings). Recommended: true'},

    'Spatial smoothing': {'type': 'subheading'},

    'spatial_smoothing': {'type': 'CheckButton', 'Recommended': 'false',
                          'Description': 'true or false. Uses SUSAN to spatial smooth. Recommended: true'},

    'smoothing_fwhm': {'type': 'Entry', 'Recommended': 8.0,
                       'label': 'Spatial smoothing fwhm (mm)',
                       'Description': 'fwhm of smoothing, in mm, gets converted using sqrt(8*log(2)). Recommended: 8.0'},

    'smoothing_brightness_threshold': {'type': 'Entry', 'Recommended': 2000.0,
                                       'label': 'Spatial smoothing brightness threshold',
                                       'Description': 'Should be greater than noise level and less than contrast of '
                                                      'edges to be preserved. Recommended: 2000.0'},

    'Image SNR calculation': {'type': 'subheading'},

    'magnitude_correction': {'type': 'CheckButton', 'Recommended': 'true',
                             'Description': 'true or false. Correction factor of 0.7 applied when running iSNR '
                                            'calculations, to correct for Rayleigh distributed noise when using '
                                            'magnitude vs complex images. \nReference: Constantinides, C. D., Atalar, '
                                            'E., & McVeigh, E. R. (1997). Signal-to-Noise Measurements in Magnitude '
                                            'Images from NMR Phased Arrays.'},

    'noise_volume': {'type': 'CheckButton', 'Recommended': 'false', 'label': 'Noise volume included in time series',
                     'Description': 'true or false. Select true if a noise volume has been collected as part of the fMRI '
                                    'time series.\n'
                                    'NOTE: If true, the noise volume in the time series will be separated from the '
                                    'functional volumes and will be placed into the folder "func_noiseVolumeRemoved".\n'
                                    'If "noise volume" is true and "noise value" is not none, the noise volume '
                                    'will be used in image SNR calculation rather than the user defined noise value.'},

    'noise_volume_location': {'type': 'OptionMenu', 'Recommended': 'End', 'Options': ['Beginning', 'End'],
                              'save_as': 'string',
                              'Description': "'max' to select number of cores available on the system, alternatively "
                                             "an int to manually select number of cores to use. Recommended: 'max'"},

    'iSNR_std_use_only_nonzero_voxels': {'type': 'CheckButton', 'Recommended': 'true',
                                         'label': 'Use only nonzero voxels for iSNR calc',
                                         'Description': 'true or false.'},

    'Add Gaussian noise': {'type': 'subheading'},

    'create_noise_level_file': {'type': 'Button', 'Command': 'create_noise_file', 'Text': 'Create noiseValues.csv',
                                'Pass self': False,
                                'Description': 'Create a noiseValues.csv file to determine what the standard deviation'
                                               "of the Gaussian should be when adding noise to each participant's data "
                                               '(the distribution will have a mean of 0).'
                                               'This will not overwrite the original files. '
                                               'Recommended: The standard deviation over time found in '
                                               'the dataset.'
                                               '\nNOTE: Can be used to see how additional noise affects '
                                               'analysis. To calculate the noise level for each participant, '
                                               'create tSNR maps with the fRAT and run the '
                                               'full analysis using the tStd files. The mean values of '
                                               'this analysis in each participants subfolder, will then show you the '
                                               'average noise for each region for each participant.'
                                               '\n\nThis file can also be used to manually set a noise value for each '
                                               'participant for use in iSNR calculation. This noise value can be '
                                               'calculated as the standard deviation of voxel values outside of the '
                                               'brain. If "Noise volume included in time series" is set to true, '
                                               'standard deviation of noise will be calculated using the noise volume, '
                                               'even if a noise value has been provided in this file.'},

    'noise_multipliers': {'type': 'Entry', 'Recommended': 1, 'save_as': 'list', 'label': 'Noise multiplier(s)',
                          'Description': "Provide a comma-separated list of multipliers for the standard deviation of "
                                         "the gaussian noise to plot e.g. '1, 5'. A separate file will be produced for each multiplier."
                                         "\nNOTE: Noise has a gaussian distribution, with a mean of 0 and a "
                                         "standard deviation of the noise level of each participant * multiplier."}
}
