from pathlib import Path

from nipype.interfaces.fsl import ImageStats, TemporalFilter, maths
import logging

from utils import *


def file_setup(func):
    if config.output_folder_name != 'DEFAULT':
        output_folder = config.output_folder_name
    elif func == 'Image SNR':
        output_folder = 'imageSNR_report'
    elif func == 'Temporal SNR':
        output_folder = 'temporalSNR_report'

    file_location = Utils.file_browser()

    # Find all nifti and analyze files
    files = Utils.find_files(f"{file_location}", "hdr", "nii.gz", "nii")

    Utils.check_and_make_dir(f"{file_location}/{output_folder}")

    return file_location, files, f"{file_location}/{output_folder}"


def load_brain(file_path):
    data = nib.load(file_path)

    header = data.header
    data = data.get_fdata()

    return data, header


def calculate_sigma_in_volumes(file_path):
    data = nib.load(file_path)
    TR = data.header['pixdim'][4]  # Find TR

    # Equation found here: https://www.jiscmail.ac.uk/cgi-bin/webadmin?A2=FSL;f6fd75a6.1709
    return 1 / (2 * config.highpass_filter_cutoff * TR)


def save_brain(data, ext, no_ext_file, output_folder, header=None):
    brain = nib.Nifti1Pair(data, None, header)
    nib.save(brain, f"{output_folder}/{no_ext_file}_{ext}.nii.gz")

    return f"{output_folder}/{no_ext_file}_{ext}.nii.gz"


def temporalSNR_calc(file, no_ext_file, output_folder):
    maths.MeanImage(in_file=file, out_file=f'{output_folder}/{no_ext_file}_tMean.nii.gz').run()  # Mean over time

    maths.StdImage(in_file=file, out_file=f'{output_folder}/{no_ext_file}_tStd.nii.gz').run()  # Standard dev over time

    # tMean / tStd
    maths.BinaryMaths(operation='div', in_file=f'{output_folder}/{no_ext_file}_tMean.nii.gz',
                      operand_file=f'{output_folder}/{no_ext_file}_tStd.nii.gz',
                      out_file=f'{output_folder}/{no_ext_file}_tSNR.nii.gz').run()

    # Threshold volume so any tSNR values above 1000 are set to 0
    maths.Threshold(in_file=f'{output_folder}/{no_ext_file}_tSNR.nii.gz', thresh=1000.0, direction='above',
                    out_file=f'{output_folder}/{no_ext_file}_tSNR.nii.gz').run()

    if config.magnitude_correction:
        magnitude_correction(f'{output_folder}/{no_ext_file}_tSNR.nii.gz')


def imageSNR_calc(func_file, noise_file, no_ext_file, output_folder):
    maths.MeanImage(in_file=func_file, out_file=f'{output_folder}/{no_ext_file}_tMean.nii.gz').run()  # Mean over time

    if config.noise_volume:
        std = ImageStats(in_file=noise_file, op_string='-S',
                         terminal_output='allatonce').run()  # Std dev of entire volume  # TODO: Use -S or -s
        noise_value = std.aggregate_outputs().get()['out_stat']

    else:
        noise_value = int(noise_file)

    # tMean / Std
    maths.BinaryMaths(operation='div', in_file=f'{output_folder}/{no_ext_file}_tMean.nii.gz',
                      operand_value=noise_value,
                      out_file=f'{output_folder}/{no_ext_file}_iSNR.nii.gz').run()

    if config.magnitude_correction:
        magnitude_correction(f'{output_folder}/{no_ext_file}_iSNR.nii.gz')


def magnitude_correction(file_name):
    maths.BinaryMaths(in_file=file_name, operation='mul', operand_value=0.7, out_file=file_name).run()


def separate_noise_from_func(file, no_ext_file, output_folder):
    data, header = load_brain(file)

    if config.noise_volume_location == 'End':
        noise_data, func_data = data[:, :, :, -1], data[:, :, :, :-1]
    elif config.noise_volume_location == 'Beginning':
        noise_data, func_data = data[:, :, :, 0], data[:, :, :, 1:]
    else:
        raise Exception('Noise volume location not valid.')

    noise_file = save_brain(noise_data, 'noise_data', no_ext_file, output_folder, header)
    func_file = save_brain(func_data, 'func_data', no_ext_file, output_folder, header)

    return func_file, noise_file


def highpass_filtering(file_path, output_folder, no_ext_file):
    sigma_in_volumes = calculate_sigma_in_volumes(file_path)

    maths.MeanImage(in_file=f'{file_path}', out_file=f'{output_folder}/{no_ext_file}_tMean.nii.gz').run()

    TemporalFilter(in_file=f'{file_path}',
                   out_file=f'{output_folder}/{no_ext_file}_filtered.nii.gz',
                   highpass_sigma=sigma_in_volumes).run()

    maths.BinaryMaths(operation='add', in_file=f'{output_folder}/{no_ext_file}_tMean.nii.gz',
                      operand_file=f'{output_folder}/{no_ext_file}_filtered.nii.gz',
                      out_file=f'{output_folder}/{no_ext_file}_filtered_restoredmean.nii.gz').run()

    return f'{output_folder}/{no_ext_file}_filtered_restoredmean.nii.gz', \
           f'{output_folder}/{no_ext_file}_filtered.nii.gz'


def delete_files(redundant_files):
    for file in redundant_files:
        os.remove(file)


def create_maps(func, file, no_ext_file, noise_file, output_folder):
    if func == 'Image SNR':
        imageSNR_calc(file, noise_file, no_ext_file, output_folder)
    elif func == 'Temporal SNR':
        temporalSNR_calc(file, no_ext_file, output_folder)


def clean_files(file, no_ext_file, output_folder):
    noise_file = config.manual_noise_value
    redundant_files = []

    if config.noise_volume:
        file, noise_file = separate_noise_from_func(file, no_ext_file, output_folder)
        redundant_files.extend([file, noise_file])

    if config.motion_correction:
        fsl.MCFLIRT(in_file=file, out_file=f'{output_folder}/{no_ext_file}_motion_corrected.nii.gz').run()
        file = f'{output_folder}/{no_ext_file}_motion_corrected.nii.gz'

    if config.temporal_filter:
        file, redundant_file = highpass_filtering(file, output_folder, no_ext_file)
        redundant_files.extend([file, redundant_file])

    if config.spatial_smoothing:
        fsl.SUSAN(in_file=file, fwhm=config.smoothing_fwhm, brightness_threshold=config.smoothing_brightness_threshold,
                  out_file=f'{output_folder}/{no_ext_file}_smoothed.nii.gz').run()
        file = f'{output_folder}/{no_ext_file}_smoothed.nii.gz'

    return file, noise_file, redundant_files


def process_files(file, file_location, output_folder, func, cfg):
    global config

    config = cfg

    no_ext_file = Utils.strip_ext(file)
    file = f"{file_location}/{file}"

    file, noise_file, redundant_files = clean_files(file, no_ext_file, output_folder)
    create_maps(func, file, no_ext_file, noise_file, output_folder)

    delete_files(redundant_files)


def main(func):
    global config

    config = Utils.load_config(Path(os.path.abspath(__file__)).parents[0], 'statmap_config.toml')  # Reload config file incase GUI has changed it
    logging.getLogger('nipype.workflow').setLevel(0)  # Suppress workflow terminal output

    if func == 'Image SNR' and not config.noise_volume and config.manual_noise_value == '':
        raise Exception('Image SNR calculation selected but "Noise volume" is not true. \n '
                        'Make sure this option is set to true and the position of the noise '
                        'volume in the fMRI data is correctly set.')

    if func == 'Image SNR' and config.noise_volume and config.manual_noise_value != '':
        warnings.warn('"Noise volume" is true and a manual noise value has also been given. Using noise volume for '
                      'image SNR calculation. If this is not correct, set "Noise volume" to false.')

    file_location, files, output_folder = file_setup(func)

    calculate_statistical_maps(files, file_location, output_folder, func)

    if config.verbose:
        print(f"{func} images created in {output_folder}.")


def calculate_statistical_maps(files, file_location, output_folder, func):
    if config.multicore_processing:
        pool = Utils.start_processing_pool()
    else:
        pool = None

    iterable = zip(files, itertools.repeat(file_location), itertools.repeat(output_folder),
                   itertools.repeat(func), itertools.repeat(config))

    if config.multicore_processing:
        pool.starmap(process_files, iterable)
    else:
        list(itertools.starmap(process_files, iterable))

    if config.multicore_processing:
        Utils.join_processing_pool(pool, restart=False)
