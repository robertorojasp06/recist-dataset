# recist-dataset
Description and analysis of a dataset for RECIST protocol.

## Download raw data
Please send an [email](mailto:roberto.rojas.pi@uchile.cl) requesting for the final data, with the subject "RECIST-dataset request".

## Steps to obtain the final data from raw data (personal usage)

1. Clone the repository and install the conda environment running `conda env create -f environment.yml`.

2. Run `get_3d_instance_annotated_lesions.py` script to convert `raw` annotated data to `final` annotated data, containing masks with individual lesion instances. See the help using the flag `-h` to
understand the input arguments.

3. Run the following scripts to get the final version of metadata files:
	- `translate_patients_csv.py`: translate the raw `patients.csv` file from spanish to english.
	- `get_final_metadata.py`: get the final version of all metadata files.

## Compute statistics from final data

1. Run the other scripts that compute statistics from final dataset:
	- `get_nifti_metadata.py`: get information about image shape and voxel resolution.
	- `get_intensity_distributions.py`: get statistics from voxel intensities.
	- `compute_lesions_features.py`: get some features from individual lesion instances (longest axis, shortest axis, volume, mean intensity in HU).

2. Update all output files used by the jupyter notebooks in the `notebooks` folder.

## Obtain sample images from final data and results

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
- `normalize_ct_images.py`: apply windowing-normalization to CT images. The `windows_mapping.json`
file is provided to properly normalize each CT image in the final dataset.

## Variable Descriptions

### Patient information (`patients.csv`)

| Variable Name     | Type        | Description                                                  | Example        |
|-------------------|-------------|--------------------------------------------------------------|----------------|
| `patient_id`      | Integer     | Pseudonymized identifier of the patient             		 | `1`       	  |
| `patient_id`      | Integer     | Subset assigned to the patient (`training`, `test`)          | `training`     |
| `first_study_date`| String      | Date of the first study (baseline) in the format `YYYYMMDD`  | `20190220`     |
| `sex`             | String 	  | Biological sex of the patient (`M`, `F`)    			     | `F`            |
| `age`             | Integer     | Age of the patient in years                                  | `57`           |
| `diagnosis`       | String      | Clinical diagnosis assigned to the patient                   | `Lung Cancer`  |
| `health_insurance`| String      | Health insurance of the patient (`public`, `private`, `uninsured`) | `public` |

### Series information (`series.json`)

| Variable Name     | Type        | Description                                                  | Example        |
|-------------------|-------------|--------------------------------------------------------------|----------------|
| `id`      		| Integer     | Pseudonymized identifier of the series                       | `1`       	  |
| `region`          | String      | Anatomical region (`abdomen`, `thorax`)    				     | `thorax`       |
| `study_id`        | Integer 	  | Pseudonymized identifier of the study    			         | `150`          |
| `study_date`      | String      | Date of the study in the format `YYYYMMDD`  			     | `20221012`     |
| `patient_id`      | Integer     | Pseudonymized identifier of the patient             		 | `10`       	  |
| `slice_thickness` | Float       | Slice thickness (in `mm`) used during CT acquisition         | `1.5`          |
| `row_spacing`     | Float       | Voxel size (in `mm`) in the row dimension (Y-axis)           | `0.740234375`  |
| `column_spacing`  | Float       | Voxel size (in `mm`) in the column dimension (X-axis)        | `0.740234375`  |
| `slice_spacing`   | Float       | Voxel size (in `mm`) in the slice dimension (Z-axis)         | `1.0`          |
| `slices`          | Integer     | Slices count         			                             | `340`          |
| `rows`            | Integer     | Rows count         										     | `512`          |
| `columns`         | Integer     | Columns count         										 | `512`          |