Statistical_maps = {
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

    'fMRI_file_location': {'type': 'Entry', 'Recommended': "", 'save_as': 'string',
                           'label': 'Base folder location',
                           'Description': 'Either the absolute location of the folder containing the subjects or blank, '
                                          'if blank then a browser window will allow you to search for the files at '
                                          'runtime.'},

    'output_folder_name': {'type': 'Entry', 'Recommended': 'statmaps', 'save_as': 'string',
                           'Description': 'Directory to save output. '
                                          'Recommended: statmaps'},

    'temporal_filter': {'type': 'CheckButton', 'Recommended': 'true',
                        'Description': 'true or false. Use a high pass filter to remove low frequency drift. '
                                       'Recommended: true'},

    'highpass_filter_cutoff': {'type': 'Entry', 'Recommended': 0.01,
                               'label': 'Highpass filter cutoff frequency (Hz)',
                               'Description': 'Highpass filter cutoff frequency converted into sigma in seconds using '
                                              'the formula 1/(2*f*TR). Recommended: 0.01'},

    'motion_correction': {'type': 'CheckButton', 'Recommended': 'false',
                          'Description': 'true or false. Use MCFLIRT to motion correct volumes '
                                         '(uses default MCFLIRT settings). Recommended: true'},

    'spatial_smoothing': {'type': 'CheckButton', 'Recommended': 'false',
                          'Description': 'true or false. Uses SUSAN to spatial smooth. Recommended: true'},

    'smoothing_fwhm': {'type': 'Entry', 'Recommended': 8.0,
                       'label': 'Spatial smoothing fwhm (mm)',
                       'Description': 'fwhm of smoothing, in mm, gets converted using sqrt(8*log(2)). Recommended: 8.0'},

    'smoothing_brightness_threshold': {'type': 'Entry', 'Recommended': 2000.0,
                                       'label': 'Spatial smoothing brightness threshold',
                                       'Description': 'Should be greater than noise level and less than contrast of '
                                                      'edges to be preserved. Recommended: 2000.0'},

    'magnitude_correction': {'type': 'CheckButton', 'Recommended': 'true',
                             'Description': 'true or false. Temporary option. Will be removed in the future.'},

    'noise_volume': {'type': 'CheckButton', 'Recommended': 'false', 'label': 'Noise volume (for image SNR)',
                     'Description': 'true or false. Select true if a noise volume has been collected as part of the fMRI '
                                    'time series.\n'
                                    'NOTE: If true, the noise volume will not be used during temporal filtering and will '
                                    'only used if image SNR statistical maps are being created.\n'
                                    'If "noise volume" is true and "noise value" is not none, the noise volume'
                                    'will be used in image SNR calculation rather than the user defined noise value'},

    'noise_volume_location': {'type': 'OptionMenu', 'Recommended': 'End', 'Options': ['Beginning', 'End'],
                              'save_as': 'string',
                              'Description': "'max' to select number of cores available on the system, alternatively an int to manually select number of cores to use. Recommended: 'max'"},

    'manual_noise_value': {'type': 'Entry', 'Recommended': "", 'save_as': 'string',
                           'label': 'Noise value (if not using noise volume)',
                           'Description': 'Noise value can be calculated as the standard deviation of voxel values '
                                          'outside of the brain. If noise volume is set to true, standard deviation of '
                                          'noise will be calculated using the noise volume, even if a noise value has '
                                          'been provided.'}
}
