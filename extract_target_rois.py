import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import matplotlib
import matplotlib.patheffects as path_effects
import concurrent.futures
from pathlib import Path
from skimage.measure import regionprops
from matplotlib_scalebar.scalebar import ScaleBar
from tqdm import tqdm

from utils import windowing
from utils.plot import overlay_segmentation, show_obb_opencv, show_ellipse_scikit
from utils.diameters import DiameterMeasurer


matplotlib.use('Agg')


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
        self.contour_thickness = 1
        self.contour_color = (0, 0, 1.0)
        self.fitted_figure_color = (0, 1.0, 0)
        self.major_axis_color = (1.0, 0.4, 0)
        self.minor_axis_color = (1.0, 0, 1.0)
        self.text_position = (0.1, 0.85) # (x, y) 0 -> min, 1 -> max
        self.text_fontsize = 30
        self.text_fontweight = 'bold'
        self.add_fitted_figures = False
        self.figsize = (5,5)
        self.max_workers = 4
        self.verbose = False
        self.diameter_method = 'obb'
        self.unconnected_strategy = 'largest-area'
        self.add_legend = False
        self.add_axes_length_text = False
        self.add_lesion_alias_text = True

    def _plot_lesion(self, ct_slice, mask_slice, lesion_diameters,
                     fitted_figures, path_to_output_file, spacing_mm=None):
        fig, ax = plt.subplots()
        ct_seg_overlay = overlay_segmentation(
            ct_slice.astype('uint8'),
            mask_slice.astype('bool'),
            color=self.contour_color,
            only_contours=True,
            contour_thickness=self.contour_thickness,
            alpha=1.0
        )
        ax.imshow(ct_seg_overlay)
        for fig_idx, fig_ in enumerate(fitted_figures):
            if lesion_diameters["method"] == 'obb':
                if self.add_fitted_figures:
                    show_obb_opencv(
                        fig_["center_row"],
                        fig_["center_col"],
                        fig_["width_px"],
                        fig_["height_px"],
                        fig_["orientation_deg_opencv"],
                        ax,
                        edgecolor=self.fitted_figure_color,
                        linewidth=2.0
                    )
                if fig_["width_px"] > fig_["height_px"]:
                    angle_major_rad = np.deg2rad(fig_["orientation_deg_opencv"])
                    dx_major = fig_["width_px"] / 2 * np.cos(angle_major_rad)
                    dy_major = fig_["width_px"] / 2 * np.sin(angle_major_rad)
                    angle_minor_rad = np.deg2rad(fig_["orientation_deg_opencv"] + 90)
                    dx_minor = fig_["height_px"] / 2 * np.cos(angle_minor_rad)
                    dy_minor = fig_["height_px"] / 2 * np.sin(angle_minor_rad)
                else:
                    angle_major_rad = np.deg2rad(fig_["orientation_deg_opencv"] + 90)
                    dx_major = fig_["height_px"] / 2 * np.cos(angle_major_rad)
                    dy_major = fig_["height_px"] / 2 * np.sin(angle_major_rad)
                    angle_minor_rad = np.deg2rad(fig_["orientation_deg_opencv"])
                    dx_minor = fig_["width_px"] / 2 * np.cos(angle_minor_rad)
                    dy_minor = fig_["width_px"] / 2 * np.sin(angle_minor_rad)                
                # Compute major axis lines
                endpoint1_major = (fig_["center_col"] - dx_major, fig_["center_row"] - dy_major)
                endpoint2_major = (fig_["center_col"] + dx_major, fig_["center_row"] + dy_major)
                # Compute minor axis
                endpoint1_minor = (fig_["center_col"] - dx_minor, fig_["center_row"] - dy_minor)
                endpoint2_minor = (fig_["center_col"] + dx_minor, fig_["center_row"] + dy_minor)
            elif lesion_diameters["method"] == 'ellipse':
                if self.add_fitted_figures:
                    show_ellipse_scikit(
                        fig_["centroid_row"],
                        fig_["centroid_col"],
                        fig_["major_axis_length_px"],
                        fig_["minor_axis_length_px"],
                        fig_["orientation_rad_scikit"],
                        ax,
                        edgecolor=self.fitted_figure_color,
                        linewidth=2.0
                    )
                orientation_rad_norm = np.deg2rad(fig_['orientation_deg_norm'])
                dx_major = fig_["major_axis_length_px"] / 2 * np.cos(orientation_rad_norm)
                dy_major = fig_["major_axis_length_px"] / 2 * np.sin(orientation_rad_norm)
                dx_minor = fig_["minor_axis_length_px"] / 2 * np.sin(orientation_rad_norm)
                dy_minor = fig_["minor_axis_length_px"] / 2 * np.cos(orientation_rad_norm)
                # Compute major and minor axes lines
                if fig_["orientation_deg_norm"] < 90:
                    endpoint1_major = (
                        fig_["centroid_col"] - dx_major,
                        fig_["centroid_row"] + dy_major
                    )
                    endpoint2_major = (
                        fig_["centroid_col"] + dx_major,
                        fig_["centroid_row"] - dy_major
                    )
                    endpoint1_minor = (
                        fig_["centroid_col"] - dx_minor,
                        fig_["centroid_row"] - dy_minor
                    )
                    endpoint2_minor = (
                        fig_["centroid_col"] + dx_minor,
                        fig_["centroid_row"] + dy_minor
                    )
                else:
                    endpoint1_major = (
                        fig_["centroid_col"] + dx_major,
                        fig_["centroid_row"] - dy_major
                    )
                    endpoint2_major = (
                        fig_["centroid_col"] - dx_major,
                        fig_["centroid_row"] + dy_major
                    )
                    endpoint1_minor = (
                        fig_["centroid_col"] - dx_minor,
                        fig_["centroid_row"] - dy_minor
                    )
                    endpoint2_minor = (
                        fig_["centroid_col"] + dx_minor,
                        fig_["centroid_row"] + dy_minor
                    )
            else:
                raise ValueError(f"'{lesion_diameters['method']}' is not allowed as method to get the major and minor axes.")
            # Plot major axis
            ax.plot(
                [endpoint1_major[0], endpoint2_major[0]],
                [endpoint1_major[1], endpoint2_major[1]],
                color=self.major_axis_color,
                linewidth=2,
                label='Major axis' if fig_idx == 0 else None
            )
            # Plot minor axis
            ax.plot(
                [endpoint1_minor[0], endpoint2_minor[0]],
                [endpoint1_minor[1], endpoint2_minor[1]],
                color=self.minor_axis_color,
                linewidth=2,
                label='Minor axis' if fig_idx == 0 else None
            )
        # Zoom to ROI
        row_center = fig_['center_row'] if lesion_diameters["method"] == 'obb' else fig_['centroid_row']
        col_center = fig_['center_col'] if lesion_diameters["method"] == 'obb' else fig_['centroid_col']
        row_min = row_center - int(self.roi_shape[0] / 2)
        row_max = row_min + self.roi_shape[0]
        col_min = col_center - int(self.roi_shape[1] / 2)
        col_max = col_min + self.roi_shape[1]
        ax.set_xlim(col_min, col_max)
        ax.set_ylim(row_min, row_max)
        ax.axis('off')
        if self.add_legend:
            ax.legend(loc='upper right')
        # Draw diameter lengths text
        if len(fitted_figures) == 1:
            orientation_text = (
                f"\nOrientation: {fig_['orientation_deg_norm']:.2f}°"
            )
        else:
            orientation_text = ''
        info_text = (
            f"Major axis: {lesion_diameters['major_axis_mm']:.2f} mm\n"
            f"Minor axis: {lesion_diameters['minor_axis_mm']:.2f} mm"
            f"{orientation_text}"
        )
        if self.add_axes_length_text:
            ax.text(
                0.05, 0.05, info_text,
                fontsize=8,
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8)
            )
        # Draw lesion alias text
        if self.add_lesion_alias_text:
            lesion_name = lesion_diameters['lesion_label_alias']
            diameter_length = lesion_diameters['minor_axis_mm'] if lesion_diameters['lesion_label_description'].split(',')[0] == 'n' else lesion_diameters['major_axis_mm']
            text_color = self.minor_axis_color if lesion_diameters['lesion_label_description'].split(',')[0] == 'n' else self.major_axis_color
            text = ax.text(
                self.text_position[0],
                self.text_position[1],
                f"{lesion_name}: {diameter_length:.2f} mm",
                color=text_color,
                fontsize=self.text_fontsize,
                fontweight=self.text_fontweight,
                transform=ax.transAxes
            )
            text.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground='black')])
        # Draw scalebar
        if spacing_mm:
            scalebar = ScaleBar(
                spacing_mm,
                units="mm",
                fixed_value=25,
                location='lower center',
                box_alpha=1,
                box_color='white',
                font_properties={"size": 18},
                color='black'
            )
            ax.add_artist(scalebar)
        fig.savefig(
            str(path_to_output_file),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close(fig)

    def _extract_from_ct(self, path_to_ct, path_to_mask, window_name=None):
        print(f"filename: {Path(path_to_ct).name}")
        ct_image = sitk.ReadImage(path_to_ct)
        spacing = ct_image.GetSpacing()
        spacing = (spacing[2], spacing[1], spacing[0])
        ct_array = sitk.GetArrayFromImage(ct_image)
        if window_name:
            ct_array = windowing.normalize_ct(
                ct_array,
                windowing.WINDOWS.get(window_name)
            )
        mask_array = sitk.GetArrayFromImage(sitk.ReadImage(path_to_mask))
        subset_df = self.targets_df[self.targets_df["filename"] == Path(path_to_ct).name]
        all_lesions = regionprops(mask_array, spacing=spacing)
        # Diameter measurer
        diameter_measurer = DiameterMeasurer(spacing=spacing)
        diameter_measurer.method = self.diameter_method
        diameter_measurer.unconnected_strategy = self.unconnected_strategy
        diameter_measurer.return_figure_params = True
        for _, lesion_row in subset_df.iterrows():
            annotated_slice = np.where(mask_array == lesion_row["lesion_label_value"], mask_array, 0)[lesion_row["major_axis_slice_idx"]]
            lesion = [item for item in all_lesions if item.label == lesion_row['lesion_label_value']][0]
            lesion_diameters, fitted_figures = diameter_measurer.compute_diameters(
                mask_array,
                lesion
            )
            lesion_diameters.update({
                'lesion_label_alias': lesion_row['lesion_label_alias'],
                'lesion_label_description': lesion_row['label_description']
            })
            path_to_output_file = Path(self.path_to_output) / f"{lesion_row['filename']}_slice_idx_{lesion_row['major_axis_slice_idx']}_label_label_{lesion_row['lesion_label_value']}.png"
            self._plot_lesion(
                ct_array[lesion_row["major_axis_slice_idx"]],
                annotated_slice,
                lesion_diameters,
                fitted_figures,
                path_to_output_file,
                spacing_mm=spacing[1]
            )

    def extract_targets(self, path_to_train_images, path_to_train_masks,
                        path_to_test_images, path_to_test_masks, path_to_output,
                        targets_df, windows_mapping):
        self.targets_df = targets_df
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
         description= """Extract 2D ROIs centered on the target lesions.
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
        default=Path.cwd() / "data" / "final" / "images" / "train" / "images",
        help="""Path to the folder containing the training images in
        nifti format."""
    )
    parser.add_argument(
        '--path_to_train_masks',
        type=str,
        default=Path.cwd() / "data" / "final" / "images" / "train" / "masks",
        help="""Path to the folder containing the training masks in
        nifti format."""
    )
    parser.add_argument(
        '--path_to_test_images',
        type=str,
        default=Path.cwd() / "data" / "final" / "images" / "test" / "images",
        help="""Path to the folder containing the testing images in
        nifti format."""
    )
    parser.add_argument(
        '--path_to_test_masks',
        type=str,
        default=Path.cwd() / "data" / "final" / "images" / "test" / "masks",
        help="""Path to the folder containing the testing masks in"
        "nifti format."""
    )
    parser.add_argument(
        '--path_to_recist_measurements',
        type=str,
        default=Path.cwd() / "data" / "final" / "recist_measurements.csv",
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
        default=Path.cwd() / 'data' / "final" / 'metadata' / 'windows_mapping.json',
		help=f"""Path to a JSON file with a dictionary containing the
		mapping between filenames and windows."""
	)
    parser.add_argument(
        '--add_fitted_figures',
        action='store_true',
        help="""Add this flag to add the fitted figure (oriented bounding
        box or ellipse, depending on the method)."""
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
    extractor.add_fitted_figures = args.add_fitted_figures
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
