import SimpleITK as sitk
import argparse
import pandas as pd
import json
import concurrent.futures
import numpy as np
from skimage.measure import label as label_objects
from skimage.measure import regionprops
from pathlib import Path
from typing import Dict
from tqdm import tqdm

from utils.diameters import DiameterMeasurer


class LesionExtractor:
    def __init__(self) -> None:
        self.filename_extension = '.nii.gz'
        self.num_processes = 4
        self.verbose = True
        self.not_reported_value = "Not Reported"

    def _get_masks_tcia_adrenal(self, path_to_masks):
        paths_to_masks = [
            item
            for item in Path(path_to_masks).rglob(f"*{self.filename_extension}")
            if "Segmentation" in item.parts[-2]
        ]
        seg_folders = list(set([
            item.parts[-2]
            for item in paths_to_masks
        ]))
        patients = set(list([
            item.parts[-4]
            for item in paths_to_masks
        ]))
        # Check one nifti mask per folder
        assert len(paths_to_masks) == len(seg_folders), "Some Segmentation folders have more than one nifti mask in Adrenal dataset."
        # Check one nifti mask per patient
        assert len(paths_to_masks) == len(patients), "Some patients have more than one nifti mask in Adrenal dataset."
        return paths_to_masks

    def _get_masks_tcia_glis(self, path_to_masks):
        paths_to_masks = [
            item
            for item in Path(path_to_masks).rglob(f"Struct_GTV{self.filename_extension}")
        ]
        patients = set(list([
            item.parts[-4]
            for item in paths_to_masks
        ]))
        # Check one nifti mask per patient
        assert len(paths_to_masks) == len(patients), "Some patients have more than one nifti mask in GLIST-RT dataset."
        return paths_to_masks

    def _get_masks_tcia_hcc(self, path_to_masks, tumor_label_value=2):
        paths_to_masks = [
            item
            for item in Path(path_to_masks).rglob(f"*{self.filename_extension}")
            if "Segmentation" in item.parts[-2]
        ]
        seg_folders = list(set([
            item.parts[-2]
            for item in paths_to_masks
        ]))
        paths_to_json = list(Path(path_to_masks).rglob('*.json'))
        patients = list(set([
            item.parts[-4]
            for item in paths_to_masks
        ]))
        # Check JSON file with 'labelID' == 2 and 'SegmentDescription' == Tumor
        assert len(paths_to_json) == len(seg_folders), "Some segmentation folders have more than one JSON file as expected HCC-TACE-Seg dataset."
        for path in paths_to_json:
            with open(path, 'r') as file:
                metadata = json.load(file)
            tumor_meta = [
                value[0]
                for value in metadata["segmentAttributes"]
                if value[0]["SegmentDescription"] == "Tumor"
            ]
            assert len(tumor_meta) == 1, f"{path} does not have 'Tumor' in 'SegmentDescription' (or has more than one instace)."
            assert tumor_meta[0]['labelID'] == tumor_label_value, f"{path} does not have 'labelID' equal to '2' as expected."
        # Get the paths to tumor masks
        paths_to_masks = [
            item
            for item in paths_to_masks
            if int(item.name.split(self.filename_extension)[0].split('-')[1]) == tumor_label_value
        ]
        # Check one segmentation folder per patient
        assert len(seg_folders) == len(patients), "Some patients have more than segmentation folder as expected in HCC-TACE-Seg dataset."
        return paths_to_masks

    def _get_masks_kits23(self, path_to_masks):
        paths_to_masks = list(Path(path_to_masks).rglob(f"segmentation{self.filename_extension}"))
        patients = list(set([
            item.parts[-2]
            for item in paths_to_masks
        ]))
        # Check one segmentation mask per patient
        assert len(paths_to_masks) == len(patients), "Some patients have more than one segmentation (or no segmentation) as expected in HCC-TACE-Seg dataset."
        return paths_to_masks

    def _get_masks_lnq23(self, path_to_masks):
        paths_to_masks = list(Path(path_to_masks).glob(f"*_seg{self.filename_extension}"))
        return paths_to_masks

    def _get_masks_tcia_lymph(self, path_to_masks):
        paths_to_masks = [
            item
            for item in Path(path_to_masks).rglob(f"*{self.filename_extension}")
            if "segmentations" in item.parts[-2]
        ]
        return paths_to_masks

    def _get_masks_tcia_nsclc(self, path_to_masks):
        paths_to_json = list(Path(path_to_masks).rglob("*.json"))
        paths_to_masks = [
            item.parent / f"{item.name.split('-')[0]}-1{self.filename_extension}"
            for item in paths_to_json
        ]
        # Check json to have only one segment with labelID equal to 1
        for path in paths_to_json:
            with open(path, 'r') as file:
                meta = json.load(file)
            assert len(meta["segmentAttributes"]) == 1, f"{path} has more than one label (or zero)."
            assert meta["segmentAttributes"][0][0]["labelID"] == 1, f"{path} has 'labelID' different from one."
        return paths_to_masks

    def _get_masks_tcia_mediastinal_lymph(self, path_to_masks):
        paths_to_masks = [
            item
            for item in Path(path_to_masks).rglob(f"*{self.filename_extension}")
            if "Annotated" in item.parts[-2]
        ]
        return paths_to_masks

    def process_mask(self, sample: Dict):
        """Process each mask to extract lesions.

        Parameters
        ----------
        sample : dict
            This dictionary has the following keys:
                'path_to_mask' : str
                'dataset_name': str
                'lesion_label_value': int
        """
        if self.verbose:
            tqdm.write(f"processing: {sample['path_to_mask']}")
        image = sitk.ReadImage(sample["path_to_mask"])
        spacing = image.GetSpacing()
        spacing = (
            spacing[2],
            spacing[1],
            spacing[0]
        )
        mask = sitk.GetArrayFromImage(image)
        if sample['lesion_label_value'] == -1:
            labeled_mask = label_objects(mask > 0)
        else:
            labeled_mask = label_objects(mask == sample['lesion_label_value'])
        objects = regionprops(
            labeled_mask,
            spacing=spacing
        )
        measurer = DiameterMeasurer(spacing)
        info = []
        for object_ in objects:
            diameters = measurer.compute_diameters(labeled_mask, object_)
            lesion_properties = {
                    "voxels_count": object_.num_pixels,
                    "slices_count": len(np.unique(object_.coords[:, 0])),
                    "volume_mm3": object_.area,
                    "volume_ml": object_.area * 1e-3
            }
            lesion_properties.update(diameters)
            if sample["dataset_name"] == "PET-CT":
                patient = Path(sample["path_to_mask"]).parts[-3]
                study = Path(sample["path_to_mask"]).parts[-2]
            elif sample["dataset_name"] in [
                "Adrenal-ACC-Ki67",
                "GLIS-RT",
                "HCC-TACE-Seg",
                "CT-Lymph-Nodes",
                "NSCLC-Radiogenomics",
                "Mediastinal-Lymph-Node-SEG"
            ]:
                patient = Path(sample["path_to_mask"]).parts[-4]
                study = Path(sample["path_to_mask"]).parts[-3]
            elif sample["dataset_name"] == "KiTS23":
                patient = Path(sample["path_to_mask"]).parts[-2]
                study = self.not_reported_value
            elif sample["dataset_name"] == "LNQ23":
                patient = Path(sample["path_to_mask"]).name.split('-')[2]
                study = self.not_reported_value
            else:
                patient = self.not_reported_value
                study = self.not_reported_value
            data_properties = {
                    "dataset": sample["dataset_name"],
                    "patient": patient,
                    "study": study,
                    "filename": Path(sample["path_to_mask"]).name
            }
            info.append({
                key: value
                for item in [data_properties, lesion_properties]
                for key, value in item.items()
            })
        return info

    def extract_lesions_info(self, dataset_name, path_to_masks,
                             lesion_label_value):
        if dataset_name == "PET-CT":
            paths_to_masks = Path(path_to_masks).rglob(f"SEG{self.filename_extension}")
        elif dataset_name == "Adrenal-ACC-Ki67":
            paths_to_masks = self._get_masks_tcia_adrenal(path_to_masks)
        elif dataset_name == "GLIS-RT":
            paths_to_masks = self._get_masks_tcia_glis(path_to_masks)
        elif dataset_name == "HCC-TACE-Seg":
            paths_to_masks = self._get_masks_tcia_hcc(path_to_masks)
        elif dataset_name == "KiTS23":
            paths_to_masks = self._get_masks_kits23(path_to_masks)
        elif dataset_name == "LNQ23":
            paths_to_masks = self._get_masks_lnq23(path_to_masks)
        elif dataset_name == "CT-Lymph-Nodes":
            paths_to_masks = self._get_masks_tcia_lymph(path_to_masks)
        elif dataset_name == "NSCLC-Radiogenomics":
            paths_to_masks = self._get_masks_tcia_nsclc(path_to_masks)
        elif dataset_name == "Mediastinal-Lymph-Node-SEG":
            paths_to_masks = self._get_masks_tcia_mediastinal_lymph(path_to_masks)
        # MSD, KiPA cases
        else:
            paths_to_masks = Path(path_to_masks).glob(f"*{self.filename_extension}")
        samples = [
            {
                "dataset_name": dataset_name,
                "path_to_mask": path,
                "lesion_label_value": lesion_label_value
            }
            for path in paths_to_masks
            if not path.name.startswith(".")
        ]
        with concurrent.futures.ThreadPoolExecutor(self.num_processes) as pool:
            results = pool.map(self.process_mask, samples)
        results = [item for sublist in results for item in sublist]
        return results


def _check_dataset_names(datasets, valid_names):
    names = [
        item["dataset_name"]
        for item in datasets
    ]
    unaccepted = [
        name
        for name in names
        if name not in valid_names
    ]
    if unaccepted:
        print(f"Not allowed dataset names: {', '.join(unaccepted)}")
        return False
    else:
        return True


def main():
    parser = argparse.ArgumentParser(
        description="""Extract information of 3d lesion instances
        from the segmentation masks of the Medical Segmentation
        Decathlon (MSD) and other datasets.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--path_to_json',
        default="other_datasets.json",
        type=str,
        help="""Path to the JSON file that specify the datasets
        to be considered. The file structure is a list of dictionaries,
        each one having the following entries: 'dataset_name' (string),
        'path_to_masks' (string), 'lesion_label_value' (int)."""
    )
    parser.add_argument(
        '--path_to_output',
        type=str,
        default=Path.cwd(),
        help="Path to the directory to save the output."
    )
    parser.add_argument(
        '--num_threads',
        type=str,
        default=8,
        help="Number of threads for multithreading."
    )
    valid_names = [
        "PET-CT",
        "Task03_Liver",
        "Task06_Lung",
        "Task07_Pancreas",
        "Task08_HepaticVessel",
        "Task10_Colon",
        "Adrenal-ACC-Ki67",
        "GLIS-RT",
        "HCC-TACE-Seg",
        "KiPA22",
        "KiTS23",
        "Mediastinal-Lymph-Node-SEG",
        "CT-Lymph-Nodes",
        "NSCLC-Radiogenomics"
    ]
    args = parser.parse_args()
    with open(args.path_to_json, 'r') as file:
        datasets = json.load(file)
    assert _check_dataset_names(datasets, valid_names), f"Accepted dataset names: {valid_names}"
    extractor = LesionExtractor()
    extractor.num_processes = args.num_threads
    lesions_info = []
    for dataset in tqdm(datasets):
        lesions_info += extractor.extract_lesions_info(
            dataset["dataset_name"],
            dataset["path_to_masks"],
            dataset["lesion_label_value"]
        )
    pd.DataFrame(lesions_info).to_csv(
        Path(args.path_to_output) / "other_datasets_lesions_info.csv",
        index=False
    )


if __name__ == "__main__":
    main()
