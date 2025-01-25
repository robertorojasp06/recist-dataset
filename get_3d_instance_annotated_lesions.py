import json
import numpy as np
import SimpleITK as sitk
import pandas as pd
import argparse
import concurrent.futures
from skimage.measure import regionprops
from skimage.measure import label as label_objects
from pathlib import Path
from scipy.stats import mode
from tqdm import tqdm


class Preprocessor:
    def __init__(self) -> None:
        self.min_size = 50
        self.min_size_units = "voxels"
        self.min_slices = 2
        self.save_non_accepted = False
        self.label_mapping = {
            'm,bazo': 'm,spleen',
            'm,costilla': 'm,rib',
            'm,higado': 'm,liver',
            'm,pancreas': 'm,pancreas',
            'm,pared abdominal': 'm,abdominal wall',
            'm,pulmon': 'm,lung',
            'm,rinon': 'm,kidney',
            'm,ovario': 'm,ovary',
            'm,suprarrenal': 'm,suprarenal',
            'n,abdomen': 'n,abdomen',
            'n,aortocava': 'n,aortocaval',
            'n,axila': 'n,axillary',
            'n,hilio hepatico': 'n,hepatic hilum',
            'n,iliaca': 'n,iliac',
            'n,inguinal': 'n,inguinal',
            'n,mediastino': 'n,mediastinum',
            'n,mesenterica': 'n,mesenteric',
            'n,pared abdominal': 'n,abdominal wall',
            'n,pelvis': 'n,pelvis',
            'n,periaortica': 'n,periaortic',
            'n,retroperitoneal': 'n,retroperitoneal',
            'n,suprarrenal': 'n,suprarenal',
            't,pulmon': 't,lung'
        }

    @property
    def min_size_units(self):
        return self._min_size_units

    @min_size_units.setter
    def min_size_units(self, value):
        assert value in ["voxels", "ml"], f"{value} is not valid. Use 'voxels' or 'ml'."
        self._min_size_units = value

    def _is_accepted(self, object_):
        # Size criteria
        if self.min_size_units == "ml":
            meets_size = True if object_.area * 1e-3  >= self.min_size else False
        elif self.min_size_units == "voxels":
            meets_size = True if object_.num_pixels >= self.min_size else False
        # Slices count criteria
        meets_slices = True if len(np.unique(object_.coords[:, 0]).tolist()) >= self.min_slices else False
        return True if meets_size and meets_slices else False

    def _get_label_description(self, mask_array, labels_mapping, object_):
        mode_value = mode(mask_array[object_.coords[:, 0], object_.coords[:, 1], object_.coords[:, 2]]).mode
        final_label = ','.join([item.strip() for item in labels_mapping.get(str(mode_value)).split(',')])
        return final_label

    def _get_axial_lengths(self, object_):
        """Return the max and min diameter lengths.
        This is the maximum and minimum of all 2d axial slices
        of the object. """

    def _preprocess_mask(self, mask):
        """Preprocess input mask. Output mask is labeled by lesion instances,
        identified by connnectivity criteria.

        Parameters
        ----------
        mask : dict
            'path_to_mask': path to input mask in nifti format.
            'path_to_labels': path to the JSON file containing the value-label mapping.
        """
        assert list(mask.keys()) == ["path_to_mask", "path_to_labels"]
        path_to_mask = mask.get("path_to_mask")
        path_to_labels = mask.get("path_to_labels")
        print(f"filename: {path_to_mask.name}")
        with open(path_to_labels, 'r') as file:
            labels_mapping = json.load(file)
        series_name = Path(path_to_mask).name.split('.nii.gz')[0]
        mask_image = sitk.ReadImage(path_to_mask)
        mask_array = sitk.GetArrayFromImage(mask_image)
        spacing = (
            mask_image.GetSpacing()[2], # slice
            mask_image.GetSpacing()[1], # row
            mask_image.GetSpacing()[0]  # column
        )
        labeled, labeled_count = label_objects(
            mask_array,
            return_num=True
        )
        props = regionprops(
            labeled,
            spacing=spacing
            )
        for object_ in props:
            object_._is_accepted = self._is_accepted(object_)
            object_._label_description = self._get_label_description(
                mask_array,
                labels_mapping,
                object_
            )
        accepted_props = [object_ for object_ in props if object_._is_accepted]
        non_accepted_props = [object_ for object_ in props if not object_._is_accepted]
        mask_results = {
            "path_to_mask": path_to_mask,
            "filename": Path(path_to_mask).name,
            "raw_3d_objects": labeled_count,
            "removed_3d_objects": len(non_accepted_props),
            "final_3d_objects": len(accepted_props)
        }
        objects_results = []
        for object_ in props:
            label_description_en = self.label_mapping.get(object_._label_description, None)
            assert label_description_en is not None, f"unexpected label: {object_._label_description}, fix it or add it to the spanish-english mapping."
            objects_results.append(
                {
                    "path_to_mask": path_to_mask,
                    "filename": Path(path_to_mask).name,
                    "removed": not object_._is_accepted,
                    "label_value": object_.label,
                    "label_description_sp": object_._label_description,
                    "label_description_en": label_description_en,
                    "size_voxels": object_.num_pixels,
                    "volume_ml": object_.area * 1e-3,
                    "slices_count": len(np.unique(object_.coords[:, 0]).tolist())
                }
            )
        # Create final mask with accepted props
        output_mask_array = np.copy(labeled).astype(mask_array.dtype)
        for object_ in non_accepted_props:
            output_mask_array[object_.coords[:,0], object_.coords[:,1], object_.coords[:,2]] = 0
        output_mask_image = sitk.GetImageFromArray(output_mask_array)
        output_mask_image.SetOrigin(mask_image.GetOrigin())
        output_mask_image.SetDirection(mask_image.GetDirection())
        output_mask_image.SetSpacing(mask_image.GetSpacing())
        sitk.WriteImage(
            output_mask_image,
            self._path_to_output_masks / Path(path_to_mask).name
        )
        # Create final labels
        output_labels_mapping = {
            str(object_.label): self.label_mapping.get(object_._label_description)
            for object_ in accepted_props
        }
        with open(self._path_to_output_labels / f"{series_name}.json", 'w') as file:
            json.dump(output_labels_mapping, file, indent=4)
        # Create final mask with non accepted props if specified
        if self.save_non_accepted:
            non_accepted_mask_array = np.zeros(labeled.shape).astype(mask_array.dtype)
            for object_ in non_accepted_props:
                non_accepted_mask_array[object_.coords[:,0], object_.coords[:,1], object_.coords[:,2]] = object_.label
            non_accepted_mask_image = sitk.GetImageFromArray(non_accepted_mask_array)
            non_accepted_mask_image.SetOrigin(mask_image.GetOrigin())
            non_accepted_mask_image.SetDirection(mask_image.GetDirection())
            non_accepted_mask_image.SetSpacing(mask_image.GetSpacing())
            sitk.WriteImage(
                non_accepted_mask_image,
                self._path_to_non_accepted / Path(path_to_mask).name
            )
        return {"mask_results": mask_results, "objects_results": objects_results}

    def preprocess_masks(self, path_to_masks, path_to_labels,
                         path_to_output, max_workers=4):
        paths_to_masks = list(Path(path_to_masks).glob('*.nii.gz'))
        assert len(paths_to_masks) > 0, "No nifti files with masks were found."
        self._path_to_output_masks = Path(path_to_output) / "masks"
        self._path_to_output_labels = Path(path_to_output) / "labels"
        self._path_to_non_accepted = Path(path_to_output) / "non-accepted-objects"
        self._path_to_output_masks.mkdir(parents=True, exist_ok=True)
        self._path_to_output_labels.mkdir(parents=True, exist_ok=True)
        if self.save_non_accepted:
            self._path_to_non_accepted.mkdir(exist_ok=True, parents=True)
        masks_dicts = [
            {
                "path_to_mask": path,
                "path_to_labels": Path(path_to_labels) / f"{path.name.split('.nii.gz')[0]}.json"
            }
            for path in paths_to_masks
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(self._preprocess_mask, masks_dicts))
        # Aggregate results
        masks_results = [
            item["mask_results"]
            for item in results
        ]
        objects_results = []        
        for item in tqdm(results):
            objects_results.extend(item["objects_results"])
        pd.DataFrame(masks_results).to_csv(
            Path(path_to_output) / "masks_results.csv",
            index=False
        )
        pd.DataFrame(objects_results).to_csv(
            Path(path_to_output) / "objects_results.csv",
            index=False
        )


def float_or_int(value):
    if type(value) is int or type is float:
        return value
    else:
        TypeError(f"Invalid value: {value} Must be float or integer.")
    

def main():
    parser = argparse.ArgumentParser(
        description="""Preprocess raw annotated lesions by removing
        small objects. Output masks contain instance segmentations,
        where each instance is a 3d object identified by connectivity.
        Original labels in spanish are translated to english.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'path_to_masks',
        type=str,
        help="""Path to the directory containing the input masks
        in nifti format (.nii.gz)."""
    )
    parser.add_argument(
        'path_to_labels',
        type=str,
        help="""Path to the directory containing the JSON files
        with the mapping between input masks values and label
        descriptions. Masks nifti files and labels JSON files
        must the share the filename (except the extension)."""
    )
    parser.add_argument(
        'path_to_output',
        type=str,
        help="Path to the directory to save output results."
    )
    parser.add_argument(
        '--max_workers',
        type=int,
        default=4,
        help="Max number of threads for multithreading."
    )
    parser.add_argument(
        '--min_size',
        type=float_or_int,
        default=50,
        help="""Minimum volume for 3d objects. Connected components
        smaller than this size are removed in the output masks."""
    )
    parser.add_argument(
        '--min_size_units',
        type=str,
        default="voxels",
        choices=["voxels", "ml"],
        help="""Volume units for minimum size."""
    )
    parser.add_argument(
        '--min_slices',
        type=int,
        default=2,
        help="Objects with less slices count are removed."
    )
    parser.add_argument(
        '--save_non_accepted',
        dest='save_non_accepted',
        action='store_true',
        help="""Add this flag to also save masks only with unaccepted
        objects (removed)."""
    )
    args = parser.parse_args()
    preprocessor = Preprocessor()
    preprocessor.save_non_accepted = args.save_non_accepted
    preprocessor.min_size = args.min_size
    preprocessor.min_size_units = args.min_size_units
    preprocessor.min_slices = args.min_slices
    preprocessor.preprocess_masks(
        args.path_to_masks,
        args.path_to_labels,
        args.path_to_output,
        args.max_workers
    )


if __name__ == "__main__":
    main()
