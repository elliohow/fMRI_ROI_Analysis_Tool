import os

from nipype.interfaces import fsl
from roianalysis.utils import Utils


class NipypeFunctions:
    def __init__(self, files, output_type):
        self.files = files
        self.output_type = output_type  # Default: "NIFTI"
        self.output_folder = "nipypeResults"
        self.interfaces = ["dti", "epi", "maths", "model", "preprocess", "utils"]

        Utils.check_and_make_dir(f"{os.getcwd()}/{self.output_folder}")

    def _user_input(self, functions, run=False, list_toolboxes=True):
        current_files = []
        while True:
            print("\n--- Available functions: ---")
            [print(f) for f in dir(functions) if not f.startswith('_') and "Output" not in f and "Input" not in f]

            if list_toolboxes:
                print("\n--- Available toolboxes: ---")
                [print(f) for f in self.interfaces]

            print("\nType the name of the operation to run on files, any additional arguments can be added to the end.")
            if current_files:
                print("To use newly created files, use the -u flag after your selection. ")
            ans = input("Alternatively press enter to stop operation: ")

            ans = ans.split()
            if len(ans) > 0 and (ans[0] in dir(functions) or ans[0] in self.interfaces) and not ans[0].startswith("_"):
                if run:
                    if "-u" in ans:
                        self.files = current_files
                        ans.remove("-u")
                    else:
                        current_files = []

                    if ans[0] in self.interfaces:
                        current_files = self._fsl_interfaces(ans[0])
                    else:
                        current_files = getattr(self, ans[0])(args=' '.join(ans[1:]))  # Run function and pass in additional args
                else:
                    return ans[0], ' '.join(ans[1:])

            elif len(ans) > 0:
                print(f"{ans[0]} not found.")

            else:  # Else statement for blank input
                break

    def _generic_routine(self, module, function, args, standargs=None, folder_name="", label="", move_files=False):
        if standargs is None:
            standargs = []

        Utils.check_and_make_dir(f"{self.output_folder}/{folder_name}")

        created_files = []
        for counter, file in enumerate(self.files):
            operation = self._setup_operation(file, module, function, args, standargs,
                                              folder_name, label, move_files)   # Setup operation
            created_files.append(self._run_operation(operation))  # Run operation

            if move_files:
                # fsl.smooth doesnt allow using inputs.out_file, so this is a workaround
                Utils.move_file(created_files[counter], os.getcwd(), f"{self.output_folder}/{folder_name}")
                created_files[counter] = f"{self.output_folder}/{folder_name}/{created_files[counter]}"
        return created_files

    def _setup_operation(self, file, module, function, args, standargs=None, folder_name="", label="", move_files=False):
        if standargs is None:
            standargs = []

        operation = getattr(module, function)()
        operation.inputs.in_file = directory + "/" + file
        operation.inputs.output_type = self.output_type

        for arg in standargs:  # Read standard arguments
            exec(arg)

        operation.inputs.args = args  # Read user defined arguments

        if not move_files:
            operation.inputs.out_file = f"{self.output_folder}/{folder_name}/" \
                                        f"{file.split('/')[-1].split('.')[0]}_{label}.nii"

        return operation

    @staticmethod
    def _run_operation(operation):
        print(operation.cmdline)
        operation.run()

        cmdline = operation.cmdline.split()
        file_location = [x for x in cmdline if x.endswith(".nii")][1]  # Find loc of created files to link functions
        return file_location

    def _fsl_interfaces(self, interface):
        fsl_interface = getattr(fsl, interface)

        ans, args = self._user_input(fsl_interface, list_toolboxes=False)  # Ask user which operation in the interface to run

        created_files = self._generic_routine(fsl_interface, ans, args=args, folder_name=ans, label=ans)

        return created_files

    def motion_correct(self, args=""):
        standargs = [
                     "operation.inputs.save_plots = True",  # Save motion time series
                     "operation.inputs.save_rms = True",  # Save displacement values
                     ]
        created_files = self._generic_routine(fsl, "MCFLIRT", args=args, standargs=standargs,
                                              folder_name="mcflirt", label="mcf")
        return created_files

    def spatial_smooth(self, args=""):
        standargs = ["operation.inputs."]

        standargs[0] += input("Do you want to use a sigma or fwhm Gaussian kernel? Type your selection as e.g. "
                              "sigma = 3.0 ")

        created_files = self._generic_routine(fsl, "Smooth", args=args, standargs=standargs,
                                              folder_name="spatial_smooth", move_files=True)

        return created_files


if __name__ == '__main__':
    directory = Utils.file_browser(chdir=True)
    files = Utils.find_files(directory, "hdr", "nii.gz", "nii")

    batch = NipypeFunctions(files, output_type="NIFTI")
    batch._user_input(NipypeFunctions, True)
