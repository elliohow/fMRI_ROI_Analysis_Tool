# Contributing to fRAT

Welcome to the fRAT repository! We're excited you're here and want to contribute.

These guidelines are designed to make it as easy as possible to get involved. If you have any questions that aren't discussed below, please let us know by opening an [issue][link_issues]!

Before you start you'll need to set up a free [GitHub][link_github] account and sign in. Here are some [instructions][link_signupinstructions].

## Making a change

We appreciate all contributions to fRAT, but those accepted fastest will follow a workflow similar to the following:

**1. Comment on an existing issue or open a new issue referencing your addition.**

This allows other members of the fRAT development team to confirm that you aren't overlapping with work that's currently underway and that everyone is on the same page with the goal of the work you're going to carry out.

[This blog][link_pushpullblog] is a nice explanation of why putting this work in up front is so useful to everyone involved.

**2. [Fork][link_fork] the [fRAT repository][link_frat] to your profile.**

This is now your own unique copy of the fRAT repository.
Changes here won't affect anyone else's work, so it's a safe space to explore edits to the code!

You can clone your fRAT repository in order to create a local copy of the code on your machine.
To install the dependencies needed for development, [Poetry][link_poetry] will need to be installed on your system.
After navigating to the fRAT directory, run `poetry install` to install the dependencies from the `poetry.lock` file.

Make sure to keep your fork up to date with the original fRAT repository.
One way to do this is to [configure a new remote named "upstream"](https://help.github.com/articles/configuring-a-remote-for-a-fork/) 
and to [sync your fork with the upstream repository][link_updateupstreamwiki].

**3. Make the changes you've discussed.**

When you are working on your changes, test frequently to ensure you are not breaking the existing code.
The test analysis can be ran using the same process as [installation checking](https://fmri-roi-analysis-tool.readthedocs.io/en/latest/installation.html#checking-installation-with-a-test-analysis).

Before pushing your changes to GitHub, run the test analysis.
If you get no errors, you're ready to submit your changes!

It's a good practice to create [a new branch](https://help.github.com/articles/about-branches/) of the repository for a new set of changes.

**4. Submit a [pull request][link_pullrequest].**

A new pull request for your changes should be created from your fork of the repository.

When opening a pull request, please use one of the following prefixes:

* **[ENH]** for enhancements
* **[FIX]** for bug fixes
* **[DOC]** for new or updated documentation
* **[STY]** for stylistic changes
* **[REF]** for refactoring existing code

Pull requests should be submitted early and often (please don't mix too many unrelated changes within one PR)!
If your pull request is not yet ready to be merged, please also include the **[WIP]** prefix (you can remove it once your PR is ready to be merged).
This tells the development team that your pull request is a "work-in-progress", and that you plan to continue working on it.

Review and discussion on new code can begin well before the work is complete, and the more discussion the better!
The development team may prefer a different path than you've outlined, so it's better to discuss it and get approval at the early stage of your work.

Once your PR is ready a member of the development team will review your changes to confirm that they can be merged into the main codebase.

## Notes for New Code

#### Catching exceptions
In general, do not catch exceptions without good reason.
For non-fatal exceptions, log the exception as a warning and add more information about what may have caused the error.

#### Testing
Bug fixes should include an example that exposes the issue.
If you're not sure what this means for your code, please ask in your pull request.

## Thank you!

You're awesome. :wave::smiley:

<br>

*&mdash; Based on contributing guidelines from the [nipype][link_nipype] project.*

[link_issues]: https://github.com/elliohow/fMRI_ROI_Analysis_Tool/issues
[link_github]: https://github.com/
[link_signupinstructions]: https://help.github.com/articles/signing-up-for-a-new-github-account
[link_frat]: https://github.com/elliohow/fMRI_ROI_Analysis_Tool
[link_pushpullblog]: https://www.igvita.com/2011/12/19/dont-push-your-pull-requests/
[link_fork]: https://help.github.com/articles/fork-a-repo/
[link_nipype]: https://github.com/nipy/nipype
[link_poetry]: https://python-poetry.org/docs/
[link_updateupstreamwiki]: https://help.github.com/articles/syncing-a-fork/
[link_pullrequest]: https://help.github.com/articles/creating-a-pull-request-from-a-fork/
