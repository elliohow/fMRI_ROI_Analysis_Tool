.. include:: links.rst

====================
Key concepts of fRAT
====================
.. contents:: :local:

.. list-table::

    * - .. figure:: images/voxelwise_example.png

           Fig 1. Representations

      - .. figure:: images/ROI_example.png

           Fig 2. Representations




The ROI analysis pipeline has 3 main steps: ROI analysis, statistical analysis and figure creation; the latter two steps
requiring the ROI analysis step to first be ran. Each step outputs a configuration log file to log what
settings were used during this step. As seen in the image below, the fRAT requires: functional volumes, anatomical
volumes, voxelwise statistical maps and optionally (but recommended for cortical ROIs) an FSL FAST segmentation.

.. image:: images/entire_process.png

