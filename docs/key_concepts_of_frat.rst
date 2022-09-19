.. include:: links.rst

====================
Key concepts of fRAT
====================
.. contents:: :local:

.. list-table::

    * - .. figure:: images/voxelwise_example.png

           A voxel-wise temporal Signal-to-Noise map created using fRAT and used as one of the inputs for the ROI
           analysis. Data is from a single subject and is displayed in native space.

      - .. figure:: images/ROI_example.png

           A region of interest map created using fRAT, showing the mean temporal Signal-to-Noise for each region.
           Data is displayed in MNI152 standard space and combines data from multiple subjects.




The ROI analysis pipeline has 3 main steps: ROI analysis, statistical analysis and figure creation; the latter two steps
requiring the ROI analysis step to first be ran. Each step outputs a configuration log file to log what
settings were used during this step. As seen in the image below, the fRAT requires: functional volumes, anatomical
volumes, voxelwise statistical maps and optionally (but recommended for cortical ROIs) an FSL FAST segmentation.

.. image:: images/entire_process.png

