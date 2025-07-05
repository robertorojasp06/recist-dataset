import SimpleITK as sitk
import numpy as np
import argparse
import json
import concurrent.futures
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from pathlib import Path
from tqdm import tqdm
from matplotlib_scalebar.scalebar import ScaleBar

from utils.diameters import DiameterMeasurer
from utils.plot import show_obb_opencv, show_ellipse_scikit, overlay_segmentation
from utils.windowing import (
    WINDOWS,
    get_windows_mapping,
    check_windows_mapping,
    normalize_ct
)


matplotlib.use('Agg')


def save_figure(ct_slice, mask_slice, lesion_diameters,
                fitted_figures, path_to_output_file, spacing_mm=None):
    fig, ax = plt.subplots()
    ct_seg_overlay = overlay_segmentation(
        ct_slice.astype('uint8'),
        mask_slice.astype('bool'),
        color=(1.0, 0, 0),
        only_contours=True,
        contour_thickness=2,
        alpha=1.0
    )
    ax.imshow(ct_seg_overlay)
    for fig_idx, fig_ in enumerate(fitted_figures):
        if lesion_diameters["method"] == 'obb':
            show_obb_opencv(
                fig_["center_row"],
                fig_["center_col"],
                fig_["width_px"],
                fig_["height_px"],
                fig_["orientation_deg_opencv"],
                ax,
                edgecolor=(0, 1, 0),
                linewidth=1.0
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
            show_ellipse_scikit(
                fig_["centroid_row"],
                fig_["centroid_col"],
                fig_["major_axis_length_px"],
                fig_["minor_axis_length_px"],
                fig_["orientation_rad_scikit"],
                ax,
                edgecolor=(0, 1, 0),
                linewidth=1.0
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
            'blue', linewidth=1, label='Major axis' if fig_idx == 0 else None
        )
        # Plot minor axis
        ax.plot(
            [endpoint1_minor[0], endpoint2_minor[0]],
            [endpoint1_minor[1], endpoint2_minor[1]],
            'yellow', linewidth=1, label='Minor axis' if fig_idx == 0 else None
        )
    ax.axis('off')
    ax.legend(loc='upper right')
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
    ax.text(
        0.05, 0.05, info_text,
        fontsize=8,
        transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=0.8)
    )
    if spacing_mm:
        scalebar = ScaleBar(
            spacing_mm,
            units="mm",
            fixed_value=10,
            fixed_units="mm",
            location='lower center',
            box_alpha=1,
            box_color='black',
            font_properties={"size": 10},
            color='white'
        )
        ax.add_artist(scalebar)
    fig.savefig(
        str(path_to_output_file),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)


class Processor:
    def __init__(self):
        self.max_workers = 8
        self.verbose = False
        self.path_to_save_fitted_axes = None
        self.windows_mapping = None
        self.diameter_method = 'obb'
        self.unconnected_strategy = 'largest-area'

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
        diameter_measurer.method = self.diameter_method
        diameter_measurer.unconnected_strategy = self.unconnected_strategy
        if self.path_to_save_fitted_axes:
            diameter_measurer.return_figure_params = True
        for lesion in tqdm(lesions, disable=not self.verbose):
            label_description = labels.get(str(lesion.label), None)
            if not label_description:
                raise ValueError(f"label value {lesion.label} is not in JSON labels of filename {Path(path_to_mask).name}")
            ct_values = ct_array[
                lesion.coords[:, 0],
                lesion.coords[:, 1],
                lesion.coords[:, 2]
            ]
            if diameter_measurer.return_figure_params:
                lesion_diameters, fitted_figures = diameter_measurer.compute_diameters(
                    mask_array,
                    lesion
                )
                if self.windows_mapping:
                    window = WINDOWS.get(self.windows_mapping.get(Path(path_to_ct).name))
                    ct_slice_to_plot = normalize_ct(
                        ct_array[lesion_diameters["major_axis_slice_idx"]],
                        window
                    )
                else:
                    ct_slice_to_plot = ct_array[lesion_diameters["major_axis_slice_idx"]]
                # Plot figures
                save_figure(
                    ct_slice_to_plot,
                    mask_array[lesion_diameters["major_axis_slice_idx"]] == lesion_diameters["label_value"],
                    lesion_diameters,
                    fitted_figures,
                    Path(self.path_to_save_fitted_axes) / f"{Path(path_to_ct).name.split('*.nii.gz')[0]}_slice_idx_{lesion_diameters['major_axis_slice_idx']}_lesion_label_{lesion_diameters['label_value']}.png",
                    spacing_mm=spacing[1]
                )
            else:
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
                    "major_axis_mm": lesion_diameters["major_axis_mm"],
                    "minor_axis_mm": lesion_diameters["minor_axis_mm"],
                    "major_axis_slice_idx": lesion_diameters["major_axis_slice_idx"],
                    "method": lesion_diameters["method"],
                    "unconnected_strategy": lesion_diameters["unconnected_strategy"]
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
        '--save_axes',
        action='store_true',
        help="Add this flag to save plots of major and minor axes."
    )
    parser.add_argument(
        '--max_workers',
        type=int,
        default=8,
        help="Max threads for multithreading."
    )
    parser.add_argument(
		'--window',
		type=str,
        default=None,
		help=f"""Window for CT normalization: {list(WINDOWS.keys())}.
		This window is applied on all CTs. Alternatively, you can provide
		the path to a JSON file with a dictionary containing the
		mapping between filenames and windows."""
	)
    parser.add_argument(
        '--diameter_method',
        type=str,
        choices=['ellipse', 'obb'],
        default='obb',
        help="Method to compute major and minor axes."
    )
    parser.add_argument(
        '--unconnected_strategy',
        type=str,
        choices=['sum', 'join', 'largest-area', 'largest-measurement'],
        default='largest-area',
        help="""Strategy to deal with axes computation when exists
        more than one connected component. 'join' is only accepted
        for 'obb' method."""
    )
    args = parser.parse_args()
    processor = Processor()
    processor.max_workers = args.max_workers
    processor.diameter_method = args.diameter_method
    processor.unconnected_strategy = args.unconnected_strategy
    if args.window:
        windows_mapping = get_windows_mapping(
            args.window,
            args.path_to_cts
        )
        check_windows_mapping(
            windows_mapping,
            args.path_to_cts
        )
        processor.windows_mapping = windows_mapping
    if args.save_axes:
        path_to_fitted = Path(args.path_to_output) / "plots-axes"
        path_to_fitted.mkdir(exist_ok=True)
        processor.path_to_save_fitted_axes = path_to_fitted
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
