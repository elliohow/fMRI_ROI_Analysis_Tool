# brain_roi_analysis_tool
Make sure that nipype version matches your version of FSL and Freesurfer

Scaled figures should only be used if all files analysed together are also displayed together, otherwise the
scaling could have been carried out based on a file which isn't being presented.

## Potential errors
### Multicore processing
On some Mac OS systems, multicore processing may cause the below issue:

`objc[16599]: +[__NSCFConstantString initialize] may have been in progress in another thread when fork()`

#### Solution
In the terminal, edit your bash_profile with `nano ~/.bash_profile`

At the end of the bash_profile file add the line: `export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES`

Then save and exit the bash_profile. Solution found here: [Link](https://stackoverflow.com/questions/50168647/multiprocessing-causes-python-to-crash-and-gives-an-error-may-have-been-in-progr)