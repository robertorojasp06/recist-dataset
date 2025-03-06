import numpy as np
import SimpleITK as sitk
from skimage.measure import label as label_objects
from skimage.measure import regionprops
from typing import Tuple
from pathlib import Path


class DiameterMeasurer:
    """Class to measure lesion longest axes.

    Parameters
    ----------
    spacing : tuple
        Voxel size in the following order: slices, rows, columns.
    """
    def __init__(self, spacing: Tuple[float, float, float]):
        self.spacing = spacing

    def compute_diameters(self, mask, lesion):
        """Compute diameters of a lesion in the specified mask.

        Parameters
        ----------
        mask : array
            Labeled mask obtained using 'skimage.measures.label'.
            Each lesion is identified with a unique foreground value.
        lesion : object
            Lesion object obtained using 'skimage.measures.regionprops'
        """
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


def measure_lesions_axes(path_to_mask, min_size=1, label_mask=False):
    """Return a list with dictionaries, one for each
    connected component, containing the major and minor
    axes (major axis is the longest axis of all axial slices).
    Units are defined by the spacing units informed in the nifti file.

    Parameters:
    -----------
    path_to_mask : str
        Path to the nifti file containing the mask.
    min_size : int
        Minimum size, in voxels, of connected components to be measured.
    label_mask : bool
        If True, label the mask by connectivity criteria. Else, the
        input mask is expected to be a labeled array with each integer
        defining a connected component.
    """
    image = sitk.ReadImage(path_to_mask)
    spacing = image.GetSpacing()
    spacing = (spacing[2], spacing[1], spacing[0])
    array = sitk.GetArrayFromImage(image)
    labeled_image = label_objects(array) if label_mask else array
    props = [
        object_
        for object_ in regionprops(labeled_image, spacing=spacing)
        if object_.num_pixels >= min_size
    ]
    measurer = DiameterMeasurer(spacing=spacing)
    output = [
        {
            "filename": Path(path_to_mask).name,
            **measurer.compute_diameters(labeled_image, object_)
        }
        for object_ in props
    ]
    return output
