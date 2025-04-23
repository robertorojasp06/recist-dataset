# recist-dataset
Description and analysis of a dataset for RECIST protocol.

## Download raw data
Please send an [email](mailto:roberto.rojas.pi@uchile.cl) requesting for the raw and/or final data, with the subject "RECIST-dataset request".

## Steps to obtain the final data

1. Clone the repository and install the conda environment running `conda env create -f environment.yml`.

2. Run `get_3d_instance_annotated_lesions.py` script to convert `raw` annotated data to `final` annotated data, containing masks with individual lesion instances. See the help using the flag `-h` to
understand the input arguments.

3. Run the following scripts to get the final version of metadata files:
	- `translate_patients_csv.py` to translate the raw `patients.csv` file from spanish to english.
	- `get_final_metadata.py` to get the final version of all metadata files.

4. Run the other scripts that compute statistics from final dataset:
	- `get_nifti_metadata.py` to get information about image shape and voxel resolution.
	- `get_intensity_distributions.py` to get statistics from voxel intensities.
	- `compute_lesions_features.py` to get some features from individual lesion instances (longest axis, shortest axis, volume, mean intensity in HU).

5. Update all output files used by the jupyter notebooks in the `notebooks` folder.

## Obtain images from results

The following scripts are intended to generate png images of
the final data or results obtained from experiments for the journal
submission:

- `extract_target_rois.py`: extract 2D ROIs centered on the target lesions, including the overlaying of diameter lengths. Diameter lengths
are computed fitting an ellipse.
- `extract_medsam_rois.py`: extract 2D ROIs of connected components overlaying the expert annotated countours (ground truth) and the contours
obtained from the MedSAM prediction.
- `get_images_nnunet_results.py`: extract renderings of the test CT images
used for automatic liver and lung tumor segmentation using the nnUNet. The
expert annotated and the predicted masks are also rendered.

## Additional scripts

- `get_lesions_info_from_other_datasets.py`: extract information of the
lesion instances included in the datasets that were used to train MedSAM.
