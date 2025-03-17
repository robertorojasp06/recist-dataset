import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import matplotlib.patheffects as path_effects
import concurrent.futures
from pathlib import Path
from skimage.measure import regionprops, find_contours
from skimage.measure import label as label_objects
from matplotlib.patches import Ellipse
from matplotlib_scalebar.scalebar import ScaleBar
from tqdm import tqdm

from utils import windowing


def overlay_segmentation(image, mask, color=(1, 1, 0), alpha=0.5):
    """
    Overlays a binary segmentation mask onto a grayscale image
    as an RGB image.

    Parameters:
    -----------
        image (ndarray): 2D grayscale image.
        mask (ndarray): 2D binary segmentation mask (same shape as image).
        color (tuple): RGB color for the mask overlay (default is yellow).
        alpha (float): Transparency of the mask overlay (0 = transparent, 1 = solid).

    Returns:
    --------
        overlay (ndarray): RGB image with segmentation overlay.
    """
    # Normalize the grayscale image to [0,1] for proper display
    image = image.astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min())

    # Convert grayscale to RGB
    overlay = np.dstack([image] * 3)  # Shape (H, W, 3)

    # Create a color mask
    color_mask = np.zeros_like(overlay)
    for i in range(3):  # Apply color to each channel
        color_mask[..., i] = color[i]

    # Blend the overlay with the image
    overlay = np.where(mask[..., None], (1 - alpha) * overlay + alpha * color_mask, overlay)

    return overlay


def get_targets(path_to_recist_measurements, path_to_train_features,
                path_to_test_features):

    features_df = pd.DataFrame()
    for path in (path_to_train_features, path_to_test_features):
        df = pd.read_csv(path)
        features_df = pd.concat([features_df, df])
    recist_df = pd.read_csv(path_to_recist_measurements)
    targets_df = recist_df.merge(
        features_df,
        left_on=["filename", "lesion_label_value"],
        right_on=["filename", "label_value"],
        how='left'
    )
    return targets_df


class ROIExtractor:
    def __init__(self):
        self.roi_shape = (100, 100)
        self.min_countour_len = 10
        self.contour_color = (0, 1, 0)
        self.ellipse_color = (1, 0, 0)
        self.major_axis_color = (1, 0, 0)
        self.minor_axis_color = (0, 0, 1)
        self.text_position = (12, 12) # (x, y)
        self.text_fontsize = 24
        self.text_fontweight = 'bold'
        self.add_individual_axes = False
        self.figsize = (5,5)
        self.max_workers = 1
        self.verbose = False
        self.displayed_diameter_alias = "displayed_diameter_length"

    def _add_axes(self, ax, output_roi, output_mask, spacing,
                  add_ellipse=False, is_adenopathy=False):
        contours = find_contours(output_mask > 0)
        labeled_mask = label_objects(output_mask > 0)
        objects_mm = regionprops(labeled_mask, spacing=spacing)
        objects_pix = regionprops(labeled_mask)
        # Draw contours
        contour_mask = np.zeros_like(output_mask)
        for contour in contours:
            contour_mask[contour[:, 0].astype('int'), contour[:, 1].astype('int')] = True
        overlaid = overlay_segmentation(
            output_roi,
            contour_mask,
            color=self.contour_color,
            alpha=0.7
        )
        ax.imshow(overlaid)
        # Draw axes for each 2d object
        major_axis_sum = 0
        minor_axis_sum = 0
        for object_mm, object_pix in zip(objects_mm, objects_pix):
            y0, x0 = object_pix.centroid[0], object_pix.centroid[1]
            if add_ellipse:
                ellipse = Ellipse(
                    xy=(x0, y0),
                    width=object_pix.major_axis_length,
                    height=object_pix.minor_axis_length,
                    angle=np.degrees(object_pix.orientation),
                    edgecolor=(1,0,0),
                    facecolor='none',
                    linewidth=2,
                    linestyle='--'
                )
                ax.add_patch(ellipse)
            # # Compute major and minor axis endpoints
            x_minor = np.array([x0 - np.cos(object_pix.orientation) * 0.5 * object_pix.axis_minor_length,
                                x0 + np.cos(object_pix.orientation) * 0.5 * object_pix.axis_minor_length])
            y_minor = np.array([y0 + np.sin(object_pix.orientation) * 0.5 * object_pix.axis_minor_length,
                                y0 - np.sin(object_pix.orientation) * 0.5 * object_pix.axis_minor_length])
            x_major = np.array([x0 + np.sin(object_pix.orientation) * 0.5 * object_pix.axis_major_length,
                                x0 - np.sin(object_pix.orientation) * 0.5 * object_pix.axis_major_length])
            y_major = np.array([y0 + np.cos(object_pix.orientation) * 0.5 * object_pix.axis_major_length,
                                y0 - np.cos(object_pix.orientation) * 0.5 * object_pix.axis_major_length])
            # Draw major and minor axes
            ax.plot(x_major, y_major, color=self.major_axis_color,  linewidth=3, label="Major Axis")
            ax.plot(x_minor, y_minor, color=self.minor_axis_color, linewidth=3, label="Minor Axis")
            # Draw text
            if len(objects_mm) > 1 and self.add_individual_axes:
                diameter_length = object_mm.minor_axis_length if is_adenopathy else object_mm.major_axis_length
                text_color = self.minor_axis_color if is_adenopathy else self.major_axis_color
                text = ax.text(
                    x0 - 10,
                    y0,
                    f"{round(diameter_length, ndigits=2)} mm",
                    color=text_color,
                    fontsize=int(0.75 * self.text_fontsize),
                    fontweight=self.text_fontweight
                )
                text.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground='black')])
            # Accumulate lengths
            major_axis_sum += object_mm.major_axis_length
            minor_axis_sum += object_mm.minor_axis_length
        return ax, major_axis_sum, minor_axis_sum, len(objects_mm)

    def _plot_lesion(self, slice_ct, slice_mask, obj_pixels, lesion_name,
                     filename, pixel_size, add_ellipse=False):
        aug_shape = (
            slice_ct.shape[0] + 2 * self.roi_shape[0],
            slice_ct.shape[1] + 2 * self.roi_shape[1],
        )
        aug_roi = np.zeros(aug_shape)
        aug_mask = np.zeros(aug_shape)
        aug_roi[self.roi_shape[0]:self.roi_shape[0] + slice_ct.shape[0],
                self.roi_shape[1]:self.roi_shape[1] + slice_ct.shape[1]
                ] = slice_ct
        aug_mask[self.roi_shape[0]:self.roi_shape[0] + slice_ct.shape[0],
                self.roi_shape[1]:self.roi_shape[1] + slice_ct.shape[1]
                ] = slice_mask
        # Compute offsets for bounding box corners to satisfy the ROI size
        delta_row = int((self.roi_shape[0] - (obj_pixels.bbox[2] - obj_pixels.bbox[0])) // 2)
        delta_col = int((self.roi_shape[1] - (obj_pixels.bbox[3] - obj_pixels.bbox[1])) // 2)
        aug_row_min = obj_pixels.bbox[0] + self.roi_shape[0] - delta_row
        aug_row_max = aug_row_min + self.roi_shape[0]
        aug_col_min = obj_pixels.bbox[1] + self.roi_shape[1] - delta_col
        aug_col_max = aug_col_min + self.roi_shape[1]
        output_roi = aug_roi[aug_row_min:aug_row_max, aug_col_min:aug_col_max]
        output_mask =aug_mask[aug_row_min:aug_row_max, aug_col_min:aug_col_max]

        fig, ax = plt.subplots(figsize=self.figsize)
        # Get label description of lesion
        lesion_description =  self.targets_df[
            (self.targets_df["filename"] == filename) &
            (self.targets_df["lesion_label_alias"] == lesion_name)
        ]["label_description"].item()
        # Draw axes
        ax, major_diameter, minor_diameter, objects_count = self._add_axes(
            ax,
            output_roi,
            output_mask,
            (pixel_size, pixel_size),
            add_ellipse,
            True if lesion_description.split(',')[0] == 'n' else False
        )
        if lesion_description.split(',')[0] == 'n':
            diameter_length = minor_diameter
            text_color = self.minor_axis_color
        else:
            diameter_length = major_diameter
            text_color = self.major_axis_color
        # Draw text with major axis in mm
        text = ax.text(
            self.text_position[0],
            self.text_position[1],
            f"{lesion_name}: {round(diameter_length, ndigits=2)} mm",
            color=text_color,
            fontsize=self.text_fontsize,
            fontweight=self.text_fontweight
        )
        text.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground='black')])
        # Draw scalebar
        scalebar = ScaleBar(
            pixel_size,
            units="mm",
            fixed_value=25,
            location='lower center',
            box_alpha=1,
            box_color='white',
            font_properties={"size": 18},
            color='black'
        )
        ax.add_artist(scalebar)
        # Save figure
        ax.set_axis_off()
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(
            Path(self.path_to_output) / f"{filename.split('.nii.gz')[0]}_lesion_{lesion_name}_{lesion_description.split(',')[0]}_{lesion_description.split(',')[1]}.png"
        )
        plt.close(fig)
        # Display difference
        diameter_axis_name = "minor_axis" if lesion_description.split(',')[0] == 'n' else "major_axis"
        ref_length = self.targets_df[
            (self.targets_df["filename"] == filename) &
            (self.targets_df["lesion_label_alias"] == lesion_name)
        ][diameter_axis_name].item()
        if self.verbose:
            print(f"\n filename: {filename}")
            print(f"ref_diameter_length: {ref_length} mm")
            print(f"computed_diameter_length: {diameter_length} mm")
            print(f"difference: {np.round(diameter_length - ref_length ,decimals=2)}")
        self.targets_df.loc[
            (self.targets_df["filename"] == filename) & (self.targets_df["lesion_label_alias"] == lesion_name),
            self.displayed_diameter_alias
        ] = diameter_length

    def _extract_from_ct(self, path_to_ct, path_to_mask, window_name=None):
        ct_image = sitk.ReadImage(path_to_ct)
        spacing = ct_image.GetSpacing()
        spacing = (
            spacing[1],
            spacing[0]
        )
        ct_array = sitk.GetArrayFromImage(ct_image)
        if window_name:
            ct_array = windowing.normalize_ct(
                ct_array,
                windowing.WINDOWS.get(window_name)
            )
        mask_array = sitk.GetArrayFromImage(sitk.ReadImage(path_to_mask))
        subset_df = self.targets_df[
            (self.targets_df["filename"] == Path(path_to_ct).name) &
            (self.targets_df["classification"] == "target")
        ]
        for _, lesion_row in subset_df.iterrows():
            annotated_slice = np.where(mask_array == lesion_row["lesion_label_value"], mask_array, 0)[lesion_row["major_axis_slice_idx"]]
            obj_pixels = regionprops(annotated_slice)[0]
            self._plot_lesion(
                ct_array[lesion_row["major_axis_slice_idx"]],
                annotated_slice,
                obj_pixels,
                lesion_row["lesion_label_alias"],
                Path(path_to_ct).name,
                spacing[0]
            )

    def extract_targets(self, path_to_train_images, path_to_train_masks,
                        path_to_test_images, path_to_test_masks, path_to_output,
                        targets_df, windows_mapping):
        self.targets_df = targets_df
        self.targets_df[self.displayed_diameter_alias] = None
        self.path_to_output = path_to_output
        paths_to_train_cts = list(Path(path_to_train_images).glob('*.nii.gz'))
        paths_to_test_cts = list(Path(path_to_test_images).glob('*.nii.gz'))
        paths_to_cts = paths_to_train_cts + paths_to_test_cts
        paths_to_train_masks = [
            Path(path_to_train_masks) / Path(path).name
            for path in paths_to_train_cts
        ]
        paths_to_test_masks = [
            Path(path_to_test_masks) / Path(path).name
            for path in paths_to_test_cts
        ]
        paths_to_masks = paths_to_train_masks + paths_to_test_masks
        window_names = [
            windows_mapping.get(path.name)
            if windows_mapping else None
            for path in paths_to_cts
        ]
        if self.max_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                executor.map(
                    self._extract_from_ct,
                    paths_to_cts,
                    paths_to_masks,
                    window_names
                )
        else:
            for path_to_ct, path_to_mask, window_name in tqdm(zip(paths_to_cts, paths_to_masks, window_names), total=len(paths_to_cts)):
                self._extract_from_ct(
                    path_to_ct,
                    path_to_mask,
                    window_name
                )


def main():
    parser = argparse.ArgumentParser(
         description= """Extract ROIs centered on a slice of the target ROIs.
         Save them as png with the overlaid measurements of the axes
         (diameter length).\nIMPORTANT: Slight differences in the displayed
         diameters lengths will exist (respect to the diameters computed
         in 'compute_lesions_features.py') if the ROI size is not enough
         to cover all the annotated pixels in the corresponding slice mask
         of a target lesion.""",
         formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
		'path_to_output',
		type=str,
		help="Path to the output directory."
	)
    parser.add_argument(
        '--path_to_train_images',
        type=str,
        default=Path.cwd() / "data" / "images" / "final" / "train" / "images",
        help="""Path to the folder containing the training images in
        nifti format."""
    )
    parser.add_argument(
        '--path_to_train_masks',
        type=str,
        default=Path.cwd() / "data" / "images" / "final" / "train" / "masks",
        help="""Path to the folder containing the training masks in
        nifti format."""
    )
    parser.add_argument(
        '--path_to_test_images',
        type=str,
        default=Path.cwd() / "data" / "images" / "final" / "test" / "images",
        help="""Path to the folder containing the testing images in
        nifti format."""
    )
    parser.add_argument(
        '--path_to_test_masks',
        type=str,
        default=Path.cwd() / "data" / "images" / "final" / "test" / "masks",
        help="""Path to the folder containing the testing masks in"
        "nifti format."""
    )
    parser.add_argument(
        '--path_to_recist_measurements',
        type=str,
        default=Path.cwd() / "notebooks" / "resources" / "recist_lesions_mapping.csv",
        help="""Path to the csv file with the mapping between target
        measurements and label values in the annotated masks."""
    )
    parser.add_argument(
        '--path_to_train_lesions_features',
        type=str,
        default=Path.cwd() / "notebooks" / "resources" / "lesions_features_train.csv",
        help="""Path to the csv file with the lesions features
        computed from the annotated training masks."""
    )
    parser.add_argument(
        '--path_to_test_lesions_features',
        type=str,
        default=Path.cwd() / "notebooks" / "resources" / "lesions_features_test.csv",
        help="""Path to the csv file with the lesions features
        computed from the annotated testing masks."""
    )
    parser.add_argument(
		'--window',
		type=str,
        default=Path.cwd() / 'data' / 'metadata' / 'windows_mapping.json',
		help=f"""Path to a JSON file with a dictionary containing the
		mapping between filenames and windows."""
	)
    parser.add_argument(
        '--add_individual_axes_lengths',
        action='store_true',
        help="""Add this flag to add the displaying of individual axes
        lengths for ROIs with more than one 2d object."""
    )
    args = parser.parse_args()
    targets_df = get_targets(
        args.path_to_recist_measurements,
        args.path_to_train_lesions_features,
        args.path_to_test_lesions_features
    )
    if args.window:
            windows_mapping = windowing.get_windows_mapping(
                args.window,
                args.path_to_train_images
            )
            windows_mapping.update(
                windowing.get_windows_mapping(
                    args.window,
                    args.path_to_test_images
                )
            )
            windowing.check_windows_mapping(
                windows_mapping,
                args.path_to_train_images
            )
            windowing.check_windows_mapping(
                windows_mapping,
                args.path_to_test_images
            )
    else:
        windows_mapping = None
    path_to_output_imgs = Path(args.path_to_output) / ("png")
    path_to_output_imgs.mkdir(parents=True, exist_ok=True)
    extractor = ROIExtractor()
    extractor.add_individual_axes = args.add_individual_axes_lengths
    extractor.extract_targets(
        args.path_to_train_images,
        args.path_to_train_masks,
        args.path_to_test_images,
        args.path_to_test_masks,
        path_to_output_imgs,
        targets_df,
        windows_mapping
    )
    extractor.targets_df.to_csv(
        Path(args.path_to_output) / "targets.csv",
        index=False
    )


if __name__ == "__main__":
    main()
