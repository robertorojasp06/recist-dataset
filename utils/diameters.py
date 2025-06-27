import numpy as np
import SimpleITK as sitk
import cv2
from skimage.measure import label as label_objects
from skimage.measure import regionprops
from typing import Tuple, Literal
from pathlib import Path


class DiameterMeasurer:
    """Class to measure lesion longest axes.

    Parameters
    ----------
    spacing : tuple
        Voxel size in the following order: slices, rows, columns.
    """
    def __init__(self, spacing: Tuple[float, float, float],
                 method: Literal['ellipse', 'obb'] = 'obb',
                 unconnected_strategy: Literal['sum', 'join', 'largest-area', 'largest-measurement'] = 'join'):
        self.spacing = spacing
        self.method = method
        self.unconnected_strategy = unconnected_strategy

    def _get_major_axis_obb(self, contour):
        _, (width, height), _ = cv2.minAreaRect(contour)
        return max(width, height)

    def _get_minor_axis_obb(self, contour):
        _, (width, height), _ = cv2.minAreaRect(contour)
        return min(width, height)

    def compute_diameters(self, mask, lesion,
                          return_figure_params=False):
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
        grouped_max_diameters = {}
        grouped_min_diameters = {}
        if self.method == 'ellipse':
            # Regions props in 2d
            for slice_idx in unique_slices:
                slice_mask = (mask[slice_idx] == lesion.label)
                labeled = label_objects(slice_mask)
                objects = regionprops(
                    labeled,
                    spacing=(self.spacing[1], self.spacing[2])
                )
                # Deal with one more than one object
                if self.unconnected_strategy == 'largest-area':
                    largest_object = max(objects, key=lambda x: x.area)
                    major_length = largest_object.axis_major_length
                    minor_length = largest_object.axis_minor_length
                elif self.unconnected_strategy == 'largest-measurement':
                    largest_object = max(objects, key=lambda x: x.axis_major_length)
                    major_length = largest_object.axis_major_length
                    minor_length = largest_object.axis_minor_length
                elif self.unconnected_strategy == 'sum':
                    major_length = sum([object_.axis_major_length for object_ in objects])
                    minor_length = sum([object_.axis_minor_length for object_ in objects])
                else:
                    raise ValueError(f"unconnected_strategy set to '{self.unconnected_strategy}' is not accepted for 'ellipse' method.")
                grouped_max_diameters.update({str(slice_idx): major_length})
                grouped_min_diameters.update({str(slice_idx): minor_length})
        elif self.method == 'obb':
            # OpenCV contours
            for slice_idx in unique_slices:
                slice_mask = (mask[slice_idx] == lesion.label).astype('uint8') * 255
                contours, _ = cv2.findContours(
                    slice_mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                contours = [item for item in contours if len(item) > 1]
                if not contours:
                    continue
                # Deal with one more than one contour
                if self.unconnected_strategy == 'join':
                    joined_countours = np.vstack(contours).squeeze()
                    _, (width, height), _ = cv2.minAreaRect(joined_countours)
                    major_length = max(width, height) * self.spacing[1]
                    minor_length = min(width, height) * self.spacing[1]
                elif self.unconnected_strategy == 'largest-area':
                    largest_contour = max(contours, key=cv2.contourArea)
                    _, (width, height), _ = cv2.minAreaRect(largest_contour)
                    major_length = max(width, height) * self.spacing[1]
                    minor_length = min(width, height) * self.spacing[1]
                elif self.unconnected_strategy == 'largest-measurement':
                    largest_contour = max(contours, key=lambda x: self._get_major_axis_obb(x))
                    _, (width, height), _ = cv2.minAreaRect(largest_contour)
                    major_length = max(width, height) * self.spacing[1]
                    minor_length = min(width, height) * self.spacing[1]
                elif self.unconnected_strategy == 'sum':
                    major_length = sum([self._get_major_axis_obb(contour) for contour in contours])
                    minor_length = sum([self._get_minor_axis_obb(contour) for contour in contours])
                else:
                    raise ValueError(f"{self.unconnected_strategy} is not accepted for parameter 'unconnected_strategy'.")
                grouped_max_diameters.update({str(slice_idx): major_length})
                grouped_min_diameters.update({str(slice_idx): minor_length})
        # Get the major axis of all lesion slices
        major_axis_slice_idx = max(grouped_max_diameters, key=grouped_max_diameters.get)
        major_axis = grouped_max_diameters[major_axis_slice_idx]
        # Get the minor axis length corresponding to the major axis
        minor_axis = grouped_min_diameters[major_axis_slice_idx]
        output = {
            "label_value": lesion.label,
            "major_axis": major_axis,
            "minor_axis": minor_axis,
            "major_axis_slice_idx": int(major_axis_slice_idx),
            "method": self.method,
            "unconnected_strategy": self.unconnected_strategy
        }
        # Return parameters of estimated figures if specified
        if return_figure_params:
            #TODO
            pass
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
