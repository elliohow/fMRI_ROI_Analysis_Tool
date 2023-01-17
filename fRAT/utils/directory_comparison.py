import filecmp
import time
import warnings


class Test_differences:
    def __init__(self, directories, verbose_errors):
        self.directories = directories
        self.verbose_errors = verbose_errors

        self.directory_comparison = None
        self.missing_files_left = 0
        self.missing_files_right = 0
        self.differences_found = 0
        self.status = ''

        self.files_to_check = (
            # Check statmap step
            'number_of_outliers_removed',
            'P1_MB3_S2_matchBW_tSNR.nii.gz',

            # Check analysis step
            'combined_results.json',
            'combined_results_ps0.json',
            'combined_results_ps1.json',
            'combined_results_ps2.json',
            'mb4_s1.0.json'
            'additional_info.csv',
            'copy_paramValues.csv',

            # Check statistics step
            'Overall_GLM_standardised_coeffs.csv',
            'Overall_LMM.csv',
            'Overall_r2.csv',
            'Occipital Pole_LMM.csv',
            'mb.png'
            'sense.png'

            # Check figure step
            'index.html',
            'Violin_plots.html',
            'Histograms_Different_xaxis.html',
            'mb4_s2.0_Sessions.png',
            'Overall_barchart.png',
            'Cuneal_Cortex_same_ylim_barchart.png'
        )

        self.warnings_setup()
        self.run_file_comparison()

    def warnings_setup(self):
        if '.DS_Store' not in filecmp.DEFAULT_IGNORES:
            filecmp.DEFAULT_IGNORES.append('.DS_Store')

        warnings.formatwarning = self.warning_on_one_line

    def run_file_comparison(self):
        print(f'\nComparing directories {self.directories[0]} & {self.directories[1]}.')
        time.sleep(0.5)  # Sleep so output does not overlap

        directory_comparison = filecmp.dircmp(self.directories[0], self.directories[1])

        self.print_differences(directory_comparison)
        self.print_final_results()

    def print_differences(self, directory_comparison):
        for name in directory_comparison.diff_files:
            if any(x in name for x in self.files_to_check):
                self.differences_found += 1

                if self.verbose_errors == 'true':
                    warnings.warn(f"Difference in file {name} found in {directory_comparison.left} and {directory_comparison.right}")

        for left_only in directory_comparison.left_only:
            self.missing_files_right += 1

            if self.verbose_errors == 'true':
                warnings.warn(f"Missing file found in {directory_comparison.right}: {left_only}")

        for right_only in directory_comparison.right_only:
            self.missing_files_left += 1

            if self.verbose_errors == 'true':
                warnings.warn(f"Missing file found in {directory_comparison.left}: {right_only}")

        for sub_folder in directory_comparison.subdirs.values():
            self.print_differences(sub_folder)

    def print_final_results(self):
        if self.differences_found == 0:
            print(f'No differences within files/folders found.')
        else:
            warnings.warn(f"{self.differences_found} differences found.")

        if self.missing_files_left == 0 and self.missing_files_right == 0:
            print(f'No missing files found.')
        else:
            if self.missing_files_left != 0:
                warnings.warn(f"{self.missing_files_left} missing files found in {self.directories[0]}.")
            else:
                print(f'No missing files found in {self.directories[0]}.')

            if self.missing_files_right != 0:
                warnings.warn(f"{self.missing_files_right} missing files found in {self.directories[1]}.")
            else:
                print(f'No missing files found in {self.directories[1]}.')

        if self.differences_found == 0 and self.missing_files_left == 0 and self.missing_files_right == 0:
            self.status = 'No errors'
        else:
            self.status = 'Errors found'

    @staticmethod
    def warning_on_one_line(message, category, filename, lineno, file):
        return f'{message}\n'

