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
                data_properties = {
                        "dataset": sample["dataset_name"],
                        "patient": Path(sample["path_to_mask"]).parts[-3],
                        "study": Path(sample["path_to_mask"]).parts[-2],
                        "filename": Path(sample["path_to_mask"]).name,
                }
            else:
                data_properties = {
                        "dataset": sample["dataset_name"],
                        "patient": self.not_reported_value,
                        "study": self.not_reported_value,
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
        Decathlon (MSD).""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'path_to_json',
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
        "Task10_Colon"
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
        Path(args.path_to_output) / "lesions_info_other_datasets.csv",
        index=False
    )


if __name__ == "__main__":
    main()
