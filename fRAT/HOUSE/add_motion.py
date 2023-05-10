from fRAT.utils import Utils

import os
import glob
import shutil
import numpy as np
import pandas as pd
from math import sin, cos
from nipype.interfaces import fsl
from nipype.interfaces.base import CommandLineInputSpec, File, TraitedSpec, traits, isdefined
from nipype.interfaces.fsl.base import FSLCommand, FSLCommandInputSpec
from os import path

UTILITY_NAME = "Add motion"
FOLDER_NAME = "added_motion"


def run(*args, **kwargs):
    add_motion_obj = AddMotion(*args, **kwargs)

    if not kwargs['return_val']:
        mean_vals = add_motion_obj.calculate_mean_motion_in_file()

        return mean_vals

    else:
        add_motion_obj.add_motion_to_file()


class ApplyXfm4DInputSpec(FSLCommandInputSpec):
    in_file = File(exists=True, position=0, argstr='%s',
                   mandatory=True, desc="timeseries to motion-correct")
    ref_vol = File(exists=True, position=1, argstr='%s',
                   mandatory=True, desc="volume with final FOV and resolution")
    out_file = File(position=2, argstr='%s',
                    genfile=True, desc="file to write", hash_files=False)
    trans_dir = File(argstr='%s', position=3,
                     desc="folder of transformation matricies")
    four_digit = traits.Bool(
        argstr='-fourdigit', desc="true mat names have four digits not five")
    interpolation = traits.Enum(
        "spline",
        "nn",
        "sinc",
        "trilinear",
        argstr="-interp %s",
        desc="interpolation method for transformation",
    )


class ApplyXfm4DOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="transform applied timeseries")


class ApplyXfm4D(FSLCommand):
    """
    Wraps the applyxfm4D command line tool for applying one 3D transform to every volume in a 4D image OR
    a directory of 3D transforms to a 4D image of the same length.
    """

    _cmd = 'applyxfm4D'
    input_spec = ApplyXfm4DInputSpec
    output_spec = ApplyXfm4DOutputSpec

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = self._gen_outfilename()
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._gen_outfilename()
        return None

    def _gen_outfilename(self):
        out_file = self.inputs.out_file
        if isdefined(out_file):
            out_file = path.realpath(out_file)
        if not isdefined(out_file) and isdefined(self.inputs.in_file):
            out_file = self._gen_fname(
                self.inputs.in_file, suffix='_warp4D')
        return path.abspath(out_file)


class AddMotion:
    def __init__(self, *args, **kwargs):
        self.config = args[0]
        self.file = args[1]
        self.no_ext_file = args[2]
        self.project_root = args[3]
        self.participant_name = args[4]
        self.participant_dir = args[5]
        self.output_folder = args[6]

        self.mean_vals = kwargs['return_val']

    def calculate_mean_motion_in_file(self):
        fsl.MCFLIRT(in_file=self.file, out_file=f"{self.participant_dir}/{self.no_ext_file}_mcf.nii.gz",
                    save_plots=True).run()

        df = pd.read_csv(f"{self.participant_dir}/{self.no_ext_file}_mcf.nii.gz.par", header=None,
                         sep="  ", engine='python')
        df = df.loc[~(df == 0).all(axis=1)]

        return list(df.abs().mean(axis=0))

    def add_motion_to_file(self):
        self.data, self.header = Utils.load_brain(self.file)
        self.mean_vals = np.array(self.mean_vals).mean(axis=1)[0]

        timepoints = self.data.shape[-1]

        # Save initial data with no added motion
        Utils.save_brain(self.data, ext=f'_motionlevel0', no_ext_file=self.no_ext_file,
                         output_folder=f"{self.participant_dir}/{self.output_folder}", header=self.header)

        fsl.maths.MeanImage(in_file=self.file, dimension='T',
                            out_file=f"{self.participant_dir}/{self.no_ext_file}_mcf_meaned.nii.gz").run()

        # Apply noise multiplier to gaussian noise
        for multiplier in self.config.motion_multipliers:
            matrix_file_directory = f"{self.participant_dir}/{self.no_ext_file}_MAT_{multiplier}"
            Utils.check_and_make_dir(matrix_file_directory)

            params = []

            for param in self.mean_vals:
                params.append(np.random.default_rng().normal(loc=0.0, scale=param * multiplier, size=timepoints))

            # Transpose matrix so values are grouped per timepoint, not per parameter
            transposed_params = np.array(params).transpose()

            for timepoint, current_params in enumerate(transposed_params):
                transformation_matrix = self.create_transformation_matrix(current_params)
                np.savetxt(f"{matrix_file_directory}/MAT_{str(timepoint).zfill(4)}", transformation_matrix)

            ApplyXfm4D(in_file=self.file, trans_dir=matrix_file_directory,
                       out_file=f"{self.participant_dir}/{self.output_folder}/{self.no_ext_file}_motionlevel{multiplier}.nii.gz",
                       ref_vol=f"{self.participant_dir}/{self.no_ext_file}_mcf_meaned.nii.gz",
                       four_digit=True,
                       interpolation="trilinear",
                       terminal_output='none').run()

        self.clean_up_files()

    @staticmethod
    def create_transformation_matrix(params):
        R_x = np.array([[1, 0, 0, 0],
                        [0, cos(params[0]), sin(params[0]), 0],
                        [0, -sin(params[0]), cos(params[0]), 0],
                        [0, 0, 0, 1]
                        ])

        R_y = np.array([[cos(params[1]), 0, -sin(params[1]), 0],
                        [0, 1, 0, 0],
                        [sin(params[1]), 0, cos(params[1]), 0],
                        [0, 0, 0, 1]
                        ])

        R_z = np.array([[cos(params[2]), sin(params[2]), 0, 0],
                        [-sin(params[2]), cos(params[2]), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]
                        ])

        T = np.array([[1, 0, 0, params[3]],
                      [0, 1, 0, params[4]],
                      [0, 0, 1, params[5]],
                      [0, 0, 0, 1]
                      ])

        return T @ R_x @ R_y @ R_z

    def clean_up_files(self):
        for mcf in glob.glob(f'{self.participant_dir}/{self.no_ext_file}_mcf*'):
            os.remove(mcf)

        for mat_dir in glob.glob(f'{self.participant_dir}/{self.no_ext_file}_MAT*'):
            shutil.rmtree(mat_dir)
