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
                 unconnected_strategy: Literal['sum', 'join', 'largest-area', 'largest-measurement'] = 'largest-area'):
        self.spacing = spacing
        self.method = method
        self.unconnected_strategy = unconnected_strategy
        self.return_figure_params = False

    def _get_major_axis_obb(self, contour):
        _, (width, height), _ = cv2.minAreaRect(contour)
        return max(width, height)

    def _get_minor_axis_obb(self, contour):
        _, (width, height), _ = cv2.minAreaRect(contour)
        return min(width, height)

    def _normalize_opencv_orientation(self, angle, width, height,
                                      output_unit: Literal['deg', 'rad'] = 'deg'):
        if width < height:
            major_axis_angle = angle + 90
        else:
            major_axis_angle = angle
        # Convert to range [0, 180), major axis relative to positive x-axis
        major_axis_angle = (180 - (major_axis_angle % 180)) % 180
        if output_unit == 'rad':
            return np.deg2rad(major_axis_angle)
        return major_axis_angle

    def _normalize_scikit_orientation(self, angle, 
                                      output_unit: Literal['deg', 'rad'] = 'deg'):
        orientation_deg = 90 + np.rad2deg(angle)
        major_axis_angle = orientation_deg % 180
        if output_unit == 'rad':
            return np.deg2rad(major_axis_angle)
        return major_axis_angle

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
                # Deal with more than one contour
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
            "major_axis_mm": major_axis,
            "minor_axis_mm": minor_axis,
            "major_axis_slice_idx": int(major_axis_slice_idx),
            "method": self.method,
            "unconnected_strategy": self.unconnected_strategy
        }
        # Return parameters of estimated figures if specified
        if self.return_figure_params:
            figures = []
            # ellipse: centroid, orientation, major axis, minor axis
            if output["method"] == 'ellipse':
                slice_mask = (mask[output["major_axis_slice_idx"]] == lesion.label)
                labeled = label_objects(slice_mask)
                objects = regionprops(
                    labeled,
                    spacing=(self.spacing[1], self.spacing[2])
                )
                if self.unconnected_strategy == 'largest-area':
                    largest_object = max(objects, key=lambda x: x.area)
                    figures.append({
                        "method": self.method,
                        "unconnected_strategy": self.unconnected_strategy,
                        "label_value": lesion.label,
                        "centroid_row": np.mean(largest_object.coords, axis=0)[0],
                        "centroid_col": np.mean(largest_object.coords, axis=0)[1],
                        "major_axis_length_px": largest_object.axis_major_length / self.spacing[1],
                        "minor_axis_length_px": largest_object.axis_minor_length / self.spacing[1],
                        "orientation_rad_scikit": largest_object.orientation,
                        "orientation_deg_norm": self._normalize_scikit_orientation(largest_object.orientation)
                    })
                elif self.unconnected_strategy == 'largest-measurement':
                    largest_object = max(objects, key=lambda x: x.axis_major_length)
                    figures.append({
                        "method": self.method,
                        "unconnected_strategy": self.unconnected_strategy,
                        "label_value": lesion.label,
                        "centroid_row": np.mean(largest_object.coords, axis=0)[0],
                        "centroid_col": np.mean(largest_object.coords, axis=0)[1],
                        "major_axis_length_px": largest_object.axis_major_length / self.spacing[1],
                        "minor_axis_length_px": largest_object.axis_minor_length / self.spacing[1],
                        "orientation_rad_scikit": largest_object.orientation,
                        "orientation_deg_norm": self._normalize_scikit_orientation(largest_object.orientation)
                    })
                elif self.unconnected_strategy == 'sum':
                    for object_ in objects:
                        figures.append({
                            "method": self.method,
                            "unconnected_strategy": self.unconnected_strategy,
                            "label_value": lesion.label,
                            "centroid_row": np.mean(object_.coords, axis=0)[0],
                            "centroid_col": np.mean(object_.coords, axis=0)[1],
                            "major_axis_length_px": object_.axis_major_length / self.spacing[1],
                            "minor_axis_length_px": object_.axis_minor_length / self.spacing[1],
                            "orientation_rad_scikit": object_.orientation,
                            "orientation_deg_norm": self._normalize_scikit_orientation(object_.orientation)
                        })
                else:
                    raise ValueError(f"unconnected_strategy set to '{self.unconnected_strategy}' is not accepted for 'ellipse' method.")
            # OBB: center, width, height, orientation
            elif output["method"] == "obb":
                slice_mask = (mask[output["major_axis_slice_idx"]] == lesion.label).astype('uint8') * 255
                contours, _ = cv2.findContours(
                    slice_mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                contours = [item for item in contours if len(item) > 1]
                # Deal with one more than one contour
                if self.unconnected_strategy == 'join':
                    joined_countours = np.vstack(contours).squeeze()
                    center, (width, height), orientation = cv2.minAreaRect(joined_countours)
                    figures.append({
                        "method": self.method,
                        "unconnected_strategy": self.unconnected_strategy,
                        "label_value": lesion.label,
                        "center_row": center[1], # center_y -> rows
                        "center_col": center[0], # center_x -> columns
                        "width_px": width,
                        "height_px": height,
                        "orientation_deg_opencv": orientation,
                        "orientation_deg_norm": self._normalize_opencv_orientation(
                            orientation,
                            width,
                            height
                        )
                    })
                elif self.unconnected_strategy == 'largest-area':
                    largest_contour = max(contours, key=cv2.contourArea)
                    center, (width, height), orientation = cv2.minAreaRect(largest_contour)
                    figures.append({
                        "method": self.method,
                        "unconnected_strategy": self.unconnected_strategy,
                        "label_value": lesion.label,
                        "center_row": center[1], # center_y -> rows
                        "center_col": center[0], # center_x -> columns
                        "width_px": width,
                        "height_px": height,
                        "orientation_deg_opencv": orientation,
                        "orientation_deg_norm": self._normalize_opencv_orientation(
                            orientation,
                            width,
                            height
                        )
                    })
                elif self.unconnected_strategy == 'largest-measurement':
                    largest_contour = max(contours, key=lambda x: self._get_major_axis_obb(x))
                    center, (width, height), orientation = cv2.minAreaRect(largest_contour)
                    figures.append({
                        "method": self.method,
                        "unconnected_strategy": self.unconnected_strategy,
                        "label_value": lesion.label,
                        "center_row": center[1], # center_y -> rows
                        "center_col": center[0], # center_x -> columns
                        "width_px": width,
                        "height_px": height,
                        "orientation_deg_opencv": orientation,
                        "orientation_deg_norm": self._normalize_opencv_orientation(
                            orientation,
                            width,
                            height
                        )
                    })
                elif self.unconnected_strategy == 'sum':
                    for contour in contours:
                        center, (width, height), orientation = cv2.minAreaRect(contour)
                        figures.append({
                            "method": self.method,
                            "unconnected_strategy": self.unconnected_strategy,
                            "label_value": lesion.label,
                            "center_row": center[1], # center_y -> rows
                            "center_col": center[0], # center_x -> columns
                            "width_px": width,
                            "height_px": height,
                            "orientation_deg_opencv": orientation,
                            "orientation_deg_norm": self._normalize_opencv_orientation(
                                orientation,
                                width,
                                height
                            )
                        })
                else:
                    raise ValueError(f"{self.unconnected_strategy} is not accepted for parameter 'unconnected_strategy'.")
            else:
                raise ValueError(f"'{output['method']}' is not allowed for 'method' parameter.")
            return output, figures
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
