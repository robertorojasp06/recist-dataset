# recist-dataset
Description and analysis of a dataset for RECIST protocol.

## Steps to replicate the final data

1. Run `get_3d_instance_annotated_lesions.py` script to convert `raw` annotated data to `final` annotated data, containing masks with individual lesion instances.  

2. Run the other scripts that compute statistics from final dataset:
	- `get_nifti_metadata.py` to get information about image shape and voxel resolution.
	- `get_intensity_distributions.py` to get statistics from voxel intensities.
	- `compute_lesions_features.py` to get some features from individual lesion instances (longest axis, shortest axis, volume, mean intensity in HU).

3. Update all output files used by the jupyter notebooks in the `notebooks` folder.
