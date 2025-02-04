import SimpleITK as sitk
import numpy as np
import argparse
import json
import concurrent.futures
import pandas as pd
from skimage.measure import regionprops
from skimage.measure import label as label_objects
from pathlib import Path
from tqdm import tqdm
from typing import Tuple


class DiameterMeasurer:
    def __init__(self, spacing: Tuple[float, float, float]):
        self.spacing = spacing

    def compute_diameters(self, mask, lesion):
        unique_slices = np.unique(lesion.coords[:, 0]).tolist()
        grouped_max_diameters = {str(idx): [] for idx in unique_slices}
        grouped_min_diameters = {str(idx): [] for idx in unique_slices}
        # Regions props in 2d
        for slice_idx in unique_slices:
            slice_mask = (mask[slice_idx] == lesion.label)
            labeled = label_objects(slice_mask)
            objects = regionprops(
                labeled,
                spacing=(self.spacing[1], self.spacing[2])
            )
            for object_ in objects:
                grouped_max_diameters[str(slice_idx)].append(object_.axis_major_length)
                grouped_min_diameters[str(slice_idx)].append(object_.axis_minor_length)
        # Compute the sum of diameters (in case of more than 1 object for slice)
        grouped_max_diameters = {
            key: np.sum(value)
            for key, value in grouped_max_diameters.items()
        }
        grouped_min_diameters = {
            key: np.sum(value)
            for key, value in grouped_min_diameters.items()
        }
        # Get the major axis of all lesion slices
        major_axis_slice_idx = max(grouped_max_diameters, key=grouped_max_diameters.get)
        major_axis = grouped_max_diameters[major_axis_slice_idx]
        # Get the minor axis length corresponding to the major axis
        minor_axis = grouped_min_diameters[major_axis_slice_idx]
        output = {
            "label_value": lesion.label,
            "major_axis": major_axis,
            "minor_axis": minor_axis,
            "major_axis_slice_idx": int(major_axis_slice_idx)
        }
        return output            

class Processor:
    def __init__(self):
        self.max_workers = 8
        self.verbose = False

    def _process_sample(self, path_to_ct, path_to_mask, path_to_labels):
        print(f"filename: {Path(path_to_ct).name}")
        lesions_features = []
        ct_image = sitk.ReadImage(path_to_ct)
        spacing = ct_image.GetSpacing()
        spacing = (spacing[2], spacing[1], spacing[0])
        ct_array = sitk.GetArrayFromImage(ct_image)
        mask_array = sitk.GetArrayFromImage(sitk.ReadImage(path_to_mask))
        lesions = regionprops(
            mask_array,
            spacing=spacing
        )
        with open(path_to_labels, 'r') as file:
            labels = json.load(file)
        assert len(lesions) == len(labels), f"regionprops objects count ({len(lesions)}) does not match objects count of JSON labels ({len(labels)}) from filename {Path(path_to_mask).name}"
        diameter_measurer = DiameterMeasurer(spacing=spacing)
        for lesion in tqdm(lesions, disable=not self.verbose):
            label_description = labels.get(str(lesion.label), None)
            if not label_description:
                raise ValueError(f"label value {lesion.label} is not in JSON labels of filename {Path(path_to_mask).name}")
            ct_values = ct_array[
                lesion.coords[:, 0],
                lesion.coords[:, 1],
                lesion.coords[:, 2]
            ]
            lesion_diameters = diameter_measurer.compute_diameters(
                mask_array,
                lesion
            )
            lesions_features.append(
                {
                    "filename": Path(path_to_ct).name,
                    "label_value": lesion.label,
                    "label_description": label_description,
                    "voxels_count": lesion.num_pixels,
                    "volume_ml": lesion.area * 1e-3,
                    "slices_count": len(np.unique(lesion.coords[:, 0]).tolist()),
                    "mean_HU": np.mean(ct_values),
                    "std_HU": np.std(ct_values),
                    "major_axis": lesion_diameters["major_axis"],
                    "minor_axis": lesion_diameters["minor_axis"],
                    "major_axis_slice_idx": lesion_diameters["major_axis_slice_idx"]
                }
            )
        return lesions_features

    def process_samples(self, path_to_cts, path_to_masks, path_to_labels):
        paths_to_cts = list(Path(path_to_cts).glob('*.nii.gz'))
        paths_to_masks = [
            Path(path_to_masks) / path.name
            for path in paths_to_cts
        ]
        paths_to_labels = [
            Path(path_to_labels) / f"{path.name.split('.nii.gz')[0]}.json"
            for path in paths_to_cts
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = executor.map(
                self._process_sample,
                paths_to_cts,
                paths_to_masks,
                paths_to_labels
            )
        return list(results)


def main():
    parser = argparse.ArgumentParser(
        description="""Script to compute features from 3d lesions.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'path_to_cts',
        type=str,
        help="Path to the directory containing CT images as nifti files."
    )
    parser.add_argument(
        'path_to_masks',
        type=str,
        help="Path to the directory containing CT masks as nifti files."
    )
    parser.add_argument(
        'path_to_labels',
        type=str,
        help="""Path to the directory containing the label mapping
        in JSON files. All files (images, masks, labels) have to share
        the same filename (without considering the file extension)."""
    )
    parser.add_argument(
        '--path_to_output',
        type=str,
        default=Path.cwd(),
        help="Path to the output directory."
    )
    parser.add_argument(
        '--max_workers',
        type=int,
        help="Max threads for multithreading."
    )
    args = parser.parse_args()
    processor = Processor()
    processor.max_workers = args.max_workers
    lesions_features = processor.process_samples(
        args.path_to_cts,
        args.path_to_masks,
        args.path_to_labels
    )
    lesions_features = [
        item
        for sublist in lesions_features
        for item in sublist
    ]
    pd.DataFrame(lesions_features).to_csv(
        Path(args.path_to_output) / "lesions_features.csv",
        index=False
    )


if __name__ == "__main__":
    main()
