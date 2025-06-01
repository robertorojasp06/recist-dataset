# recist-dataset
Description and analysis of a dataset for RECIST protocol.

## Set up the repository
1. Clone the repository and install the conda environment running `conda env create -f environment.yml`.
2. Run `conda activate recist-dataset`

## Prepare the data downloaded from TCIA
1. Download the data from this link (TODO), where you can find the following resources:
	- `SEG` data corresponds to the segmentation masks in nifti format, and the corresponding JSON files specifying the label for each annotated lesion. Data is splitted in train and test sets following this folder structure:
	<pre><code>
	📁 SEG
	├── 📁 train
	│   ├── 📁 labels
	│   │   ├── case001.nii.gz
	│   │   ├── case002.nii.gz
	│   │   └── ...
	│   ├── 📁 masks
	│   │   ├── case001_mask.nii.gz
	│   │   ├── case002_mask.nii.gz
	│   │   └── ...
	├── 📁 test
	│   ├── 📁 labels
	│   │   ├── case101.nii.gz
	│   │   ├── case102.nii.gz
	│   │   └── ...
	│   ├── 📁 masks
	│   │   ├── case101_mask.nii.gz
	│   │   ├── case102_mask.nii.gz
	│   │   └── ...
	</code></pre>
	- `CT` data corresponds to the CT series in DICOM format. Data is splitted in train and test sets following this folder structure:
	<pre><code>
	📁 CT
	├── 📁 train
	│   ├── 📁 images
	│   │   ├── case001.nii.gz
	│   │   ├── case002.nii.gz
	│   │   └── ...
	├── 📁 test
	│   ├── 📁 images
	│   │   ├── case101.nii.gz
	│   │   ├── case102.nii.gz
	│   │   └── ...
	</code></pre>
	- `recist_measurements.csv` contains the measurements of the target lesions reported using RECIST 1.1, along with the corresponding foreground label value in the segmentation mask.
	- `patients.csv` contains demographic and health information of patients.
	- `series.json` contains DICOM and Orthanc metadata for each CT series.
	- `windows_mapping.json` contains the mapping between the filenames of CT series and the corresponding window name (lung, abdomen, mediastinum). The actual window parameters can be found in `utils/windowing.py`. Run the `normalize_ct_images.py` script to apply windowing normalization on CT images in nifti format.

## Variable Descriptions
TODO
### RECIST measurements (`recist_measurements.csv`)
TODO

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


## Personal Usage

### Compute statistics

1. Run the other scripts that compute statistics from final dataset:
	- `get_nifti_metadata.py`: get information about image shape and voxel resolution.
	- `get_intensity_distributions.py`: get statistics from voxel intensities.
	- `compute_lesions_features.py`: get some features from individual lesion instances (longest axis, shortest axis, volume, mean intensity in HU).

2. Update all output files used by the jupyter notebooks in the `notebooks` folder.

### Obtain sample images

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

### Additional scripts

- `get_lesions_info_from_other_datasets.py`: extract information of the
lesion instances included in the datasets that were used to train MedSAM.
- `normalize_ct_images.py`: apply windowing-normalization to CT images. The `windows_mapping.json`
file is provided to properly normalize each CT image in the final dataset.


### How to get raw data

#### Images
1. Go to the [FONDEF repository](https://github.com/covasquezv/FONDEF_ID23I10337/tree/dev/hcuch-data).
2. Download the CT series using the `get_data_from_orthanc.py` script with the flag `--format` to choose DICOM.
3. Unzip the resulting files using the `unzip_dicom_files.py` script.

#### Masks
1. Download the folder with annotations from [this link](https://sasiba.uchile.cl/index.php/apps/files/?dir=/2023_Fondef_ID23I10337/2023_2024_Fondef_ID23I10337_go/2025_Paper/data/source&fileid=23966378).
2. Open a terminal and go to the Slicer folder.
3. Get the segmentations and CT images as nifti files:

```./Slicer --no-main-window --testing --python-script path_to_script path_to_subset_segmentations path_to_series path_to_output --suffix corrected```

where `path_to_script` is the path to `get_segmentations_as_nifti.py` script in the [FONDEF repository](https://github.com/covasquezv/FONDEF_ID23I10337/tree/dev/slicer), `path_to_subset_segmentations` is the path to the folder containing train-val or test annotations, `path_to_series` is the path to the `series.json` after DICOM downloading, and `path_to_output` is the path to the output folder. Note that you apply this script twice to have train and test sets in different folders.

#### Build the raw folder
Create the following folder structure by using the images and masks obtained in the previous steps:
	<pre><code>
	📁 raw
	├── 📁 images
	│   ├── 📁 dicom
	│   │   ├── 📁 1/
	│   │   │   ├── 📁 3051489/
	│   │   │   │   ├── 📁 Portal   5.0  I30f  1/
	│   │   │   │   │   ├── CT000000.dcm
	│   │   │   │   │   ├── CT000001.dcm
	│   │   │   │   │   └── ...
	│   │   │   └── ...
	│   │   └── ...
	│   └── 📁 nifti
	│       ├── 📁 train
	│       │   ├── 📁 images
	│       │   │   ├── case001.nii.gz
	│       │   │   ├── case002.nii.gz
	│       │   │   └── ...
	│       │   ├── 📁 labels
	│       │   │   ├── case001.json
	│       │   │   ├── case002.json
	│       │   │   └── ...
	│       │   ├── 📁 masks
	│       │   │   ├── case001.nii.gz
	│       │   │   ├── case002.nii.gz
	│       │   │   └── ...
	│       └── 📁 test
	│           ├── 📁 images
	│           ├── 📁 labels
	│           ├── 📁 masks
	├── 📁 metadata
	│   ├── patients.csv
	│   ├── series.json
	│   └── windows_mapping.json
	└── recist_measurements.csv
	</code></pre>

Some notes:
- The files inside the `dicom` folder are the obtained in the `Images` subsection.
- The files inside the `nifti` folder are the obtained in the `Masks` subsection.
- The filenames of images, labels and masks are based on the DICOM tag “Series Instance UID” `(0020,000E)`. We ommitted that from the visual representation for simplicity.
- The `recist_measurements.csv` file needs to be downloaded from [here](https://sasiba.uchile.cl/index.php/apps/files/?dir=/2023_Fondef_ID23I10337/2023_2024_Fondef_ID23I10337_go/2025_Paper/data/source&fileid=23966378)
- Regarding metadata files:
	- `patients.csv` needs to be downloaded from [here](https://sasiba.uchile.cl/index.php/apps/files/?dir=/2023_Fondef_ID23I10337/2023_2024_Fondef_ID23I10337_go/2025_Paper/data/source&fileid=23966378).
	- `series.json` was obtained in the `Images` subsection using the `get_data_from_orthanc.py` script.
	- `windows_mapping.json` is obtained using the `create_windows_mapping.py` from the [FONDEF repository](https://github.com/covasquezv/FONDEF_ID23I10337/tree/dev/hcuch-data).

#### Replace NIfTI CT images
1. Convert DICOM files of CT series into compressed nifti files using the `convert_dicom_to_nifti.py` script.
2. TODO


### Steps to obtain the final data from raw data

1. Run `get_3d_instance_annotated_lesions.py` script to convert `raw` annotated data to `final` annotated data, containing masks with individual lesion instances. See the help using the flag `-h` to
understand the input arguments.
2. Run the `get_final_metadata.py` script to get the final version of all metadata files.
3. Copy the `raw/images/nifti/train/images` folder containing the CT series in nifti format to its corresponding location in the `final` folder (do this for the train and test subsets).
4. Make sure to have the following folder structure:
	<pre><code>
	📁 final
	├── 📁 images
	│   ├── 📁 train
	│   |   ├── 📁 images
	│   |   │   ├── case001.nii.gz
	│   |   │   ├── case002.nii.gz
	│   |   │   └── ...
	│   |   ├── 📁 labels
	│   |   │   ├── case001.json
	│   |   │   ├── case002.json
	│   |   │   └── ...
	│   |   ├── 📁 masks
	│   |   │   ├── case001.nii.gz
	│   |   │   ├── case002.nii.gz
	│   |   │   └── ...
	│   └── 📁 test
	│       ├── 📁 images
	│       ├── 📁 labels
	│       ├── 📁 masks
	├── 📁 metadata
	│   ├── patients.csv
	│   ├── series.json
	│   └── windows_mapping.json
	└── recist_measurements.csv
	</code></pre>