import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import json
import argparse
from pathlib import Path
from vedo import Volume, Plotter
from tqdm import tqdm
from skimage.measure import find_contours
from scipy.ndimage import binary_dilation
from matplotlib_scalebar.scalebar import ScaleBar

from utils.plot import overlay_masks


class Visualizer:
    def __init__(self, path_to_output):
        self.path_to_output = path_to_output
        self.vol_mesh_color = 'ivory'
        self.vol_mesh_alpha = 0.25
        self.vol_mesh_lighting = 'glossy'
        self.vol_mesh_bbox_xmin_delta = 20
        self.vol_mesh_bbox_xmax_delta = 20
        self.vol_mesh_bbox_ymin_delta = 70
        self.vol_mesh_bbox_ymax_delta = 70
        self.vol_mesh_bbox_zmin_delta = 20
        self.vol_mesh_bbox_zmax_delta = 20
        self.vol_mesh_isosurface_value = None
        self.mask_mesh_color_gt = (1, 0, 0) # (1, 0, 0), (0.1, 0.52, 1.0)
        self.mask_mesh_color_prediction = (255/255, 215/255, 0/255) # (1.0, 0.84, 0), (0.83, 0.07, 0.35)
        self.mask_mesh_alpha = 0.95
        self.mask_mesh_lighting = 'plastic'
        self.mask_mesh_smooth_niter = 15
        self.mask_mesh_smooth_boundary = True
        self.output_image_size =(512, 512)
        self.zoom_factor = 1.35
        self.screenshot_scale = 2
        self.background_color = 'black'
        self.interactive = False
        self.axes = 0
        self.window_level = None
        self.window_width = None
        self.azimuth = 0
        self.elevation = 90
        self.roll = 0
        self.slice_figsize = (8, 8)
        self.slice_dpi = 300
        self.slice_prediction_color = self.mask_mesh_color_prediction
        self.slice_gt_color = self.mask_mesh_color_gt
        self.slice_alpha = 0.5
        self.slice_dilation_iters = None
        self.slice_add_scalebar = True
        self.slice_scalebar_fontsize = 60

    @property
    def path_to_output_vols(self):
        return Path(self.path_to_output) / "ct"

    @property
    def path_to_output_gt_overlaid(self):
        return Path(self.path_to_output) / "ct-gt-overlaid"

    @property
    def path_to_output_prediction_overlaid(self):
        return Path(self.path_to_output) / "ct-prediction-overlaid"

    @property
    def path_to_output_slice_overlaid(self):
        return Path(self.path_to_output) / "slice-all-overlaid"

    def render(self, path_to_ct, path_to_gt_mask, path_to_prediction_mask):
        for path in (
            self.path_to_output_vols,
            self.path_to_output_gt_overlaid,
            self.path_to_output_prediction_overlaid
        ):
            Path(path).mkdir(exist_ok=True, parents=True)
        # CT Volume as a mesh
        vol = Volume(path_to_ct)
        vol_mesh = vol.isosurface(self.vol_mesh_isosurface_value)
        vol_mesh.smooth(niter=50)
        vol_mesh.c(self.vol_mesh_color).alpha(self.vol_mesh_alpha).lighting(self.vol_mesh_lighting)
        # Shrink the bounding box a bit
        xmin, xmax, ymin, ymax, zmin, zmax = vol_mesh.bounds()
        vol_mesh.cut_with_box([
            xmin + self.vol_mesh_bbox_xmin_delta, xmax - self.vol_mesh_bbox_xmax_delta,
            ymin + self.vol_mesh_bbox_ymin_delta, ymax - self.vol_mesh_bbox_ymax_delta,
            zmin + self.vol_mesh_bbox_zmin_delta, zmax - self.vol_mesh_bbox_zmax_delta
        ])
        vol_mesh = vol_mesh.extract_largest_region()
        # Mask Volume as a mesh
        mask_meshes = {
            "gt": {
                "path": path_to_gt_mask,
                "path_to_output_folder": self.path_to_output_gt_overlaid,
                "color": self.mask_mesh_color_gt,
                "mesh": None
            },
            "prediction": {
                "path": path_to_prediction_mask,
                "path_to_output_folder": self.path_to_output_prediction_overlaid,
                "color": self.mask_mesh_color_prediction,
                "mesh": None
            }
        }
        for mask_key, mask_value in mask_meshes.items():
            mask_meshes[mask_key]["mesh"] = Volume(mask_value["path"]).isosurface(0.5)
            mask_meshes[mask_key]["mesh"].c(mask_value["color"])
            mask_meshes[mask_key]["mesh"].alpha(self.mask_mesh_alpha)
            mask_meshes[mask_key]["mesh"].lighting(self.mask_mesh_lighting)
            mask_meshes[mask_key]["mesh"].smooth(
                niter=self.mask_mesh_smooth_niter,
                boundary=self.mask_mesh_smooth_boundary
            )
        # Define common rendering parameters
        cam = {
            "elevation": self.elevation,
            "azimuth": self.azimuth,
            "roll": self.roll,
            "axes": self.axes,
            "zoom": True,
            "size": self.output_image_size,
            "interactive": self.interactive
        }
        # Show the CT mesh alone
        plotter = Plotter(
            bg=self.background_color,
            offscreen=True if not self.interactive else False
        )
        plotter.show(vol_mesh, **cam)
        plotter.zoom(self.zoom_factor)
        plotter.screenshot(
            Path(self.path_to_output_vols) / f"{Path(path_to_ct).name.split('.nii.gz')[0]}.png",
            scale=self.screenshot_scale
        )
        plotter.close()
        # Show the CT mesh overlaid with masks
        for mask_value in mask_meshes.values():
            plotter = Plotter(
                bg=self.background_color,
                offscreen=True if not self.interactive else False
            )
            plotter.show(vol_mesh, mask_value["mesh"], **cam)
            plotter.zoom(self.zoom_factor)
            plotter.screenshot(
                Path(mask_value["path_to_output_folder"]) / f"{Path(path_to_ct).name.split('.nii.gz')[0]}.png",
                scale=self.screenshot_scale
            )
            plotter.close()

    def plot_slice(self, path_to_ct, path_to_gt_mask,
                   path_to_prediction_mask, slice_idx,
                   fill_gt_mask=False, fill_predicted_mask=False):
        ct_image = sitk.ReadImage(path_to_ct)
        ct_array = sitk.GetArrayFromImage(ct_image)[slice_idx]
        gt_mask = sitk.GetArrayFromImage(sitk.ReadImage(path_to_gt_mask))[slice_idx]
        predicted_mask = sitk.GetArrayFromImage(sitk.ReadImage(path_to_prediction_mask))[slice_idx]
        # Get contour masks
        if not fill_gt_mask:
            gt_contours = find_contours(gt_mask > 0)
            gt_mask = np.zeros_like(gt_mask)
            for contour in gt_contours:
                gt_mask[contour[:, 0].astype('int'), contour[:, 1].astype('int')] = True
            if self.slice_dilation_iters:
                gt_mask = binary_dilation(
                    gt_mask,
                    iterations=self.slice_dilation_iters
                )
        else:
            gt_mask = gt_mask.astype('bool')
        if not fill_predicted_mask:
            predicted_contours = find_contours(predicted_mask > 0)
            predicted_mask = np.zeros_like(predicted_mask)
            for contour in predicted_contours:
                predicted_mask[contour[:, 0].astype('int'), contour[:, 1].astype('int')] = True
            if self.slice_dilation_iters:
                predicted_mask = binary_dilation(
                    predicted_mask,
                    iterations=self.slice_dilation_iters
                )
        else:
            predicted_mask = predicted_mask.astype('bool')
        # Plot slice with overlaid mask
        overlay = overlay_masks(
            ct_array,
            predicted_mask,
            gt_mask,
            predicted_color=self.slice_prediction_color,
            gt_color=self.slice_gt_color,
            alpha=self.slice_alpha
        )
        fig, ax = plt.subplots(figsize=self.slice_figsize)
        ax.imshow(overlay)
        ax.set_axis_off()
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.path_to_output_slice_overlaid.mkdir(
            exist_ok=True,
            parents=True
        )
        # Draw scalebar
        scalebar = ScaleBar(
            ct_image.GetSpacing()[1],
            units="mm",
            fixed_value=10,
            fixed_units="cm",
            location='lower center',
            box_alpha=1,
            box_color='black',
            font_properties={"size": self.slice_scalebar_fontsize},
            color='white'
        )
        ax.add_artist(scalebar)
        fig.savefig(
            self.path_to_output_slice_overlaid / f"{Path(path_to_ct).name.split('.nii.gz')[0]}.png",
            transparent=True,
            dpi=self.slice_dpi
        )
        plt.close(fig)


def render_sample(path_to_ct, path_to_gt, path_to_prediction,
                  path_to_output, params,
                  gt_color=None, prediction_color=None):
    visualizer = Visualizer(path_to_output)
    if gt_color:
        visualizer.mask_mesh_color_gt = gt_color
        visualizer.slice_gt_color = gt_color
    if prediction_color:
        visualizer.mask_mesh_color_prediction = prediction_color
        visualizer.slice_prediction_color = prediction_color
    slice_idx = None
    fill_gt_mask = False
    fill_predicted_mask = False
    if params:
        for key, value in params.items():
            if hasattr(visualizer, key):
                setattr(visualizer, key, value)
        slice_idx = params.get("slice_idx", slice_idx)
        fill_gt_mask = params.get("fill_mask", fill_gt_mask)
        fill_predicted_mask = params.get("fill_predicted_mask", fill_predicted_mask)
    visualizer.render(
        path_to_ct,
        path_to_gt,
        path_to_prediction
    )
    if slice_idx:
        visualizer.plot_slice(
            path_to_ct,
            path_to_gt,
            path_to_prediction,
            slice_idx,
            fill_gt_mask=fill_gt_mask,
            fill_predicted_mask=fill_predicted_mask
        )


def render_all(path_to_cts, path_to_gts, path_to_predictions,
               path_to_output, path_to_params,
               gt_color=None, prediction_color=None):
    # Read data
    paths_to_gts = sorted(list(Path(path_to_gts).glob('*.nii.gz')))
    paths_to_predictions = sorted(list(Path(path_to_predictions).glob('*.nii.gz')))
    paths_to_cts = sorted(
        [
            Path(path_to_cts) / f"{path.name.split('.nii.gz')[0]}_0000.nii.gz"
            for path in paths_to_gts
        ]
    )
    # Get custom parameters
    with open(path_to_params, 'r') as file:
        params_list = json.load(file)
    # Loop over individual CT images
    for path_to_ct, path_to_gt, path_to_prediction in tqdm(
        zip(
            paths_to_cts,
            paths_to_gts,
            paths_to_predictions
        ),
        total=len(paths_to_cts)
    ):
        tqdm.write(f"ct: {Path(path_to_ct).name}")
        params = [
            item
            for item in params_list
            if item['filename'] == Path(path_to_gt).name
        ]
        render_sample(
            str(path_to_ct),
            str(path_to_gt),
            str(path_to_prediction),
            str(path_to_output),
            params[0] if params else None,
            gt_color,
            prediction_color
        )


def rgb_tuple(s):
    try:
        parts = tuple(map(float, [item.strip() for item in s.split(',')]))
        if len(parts) != 3:
            raise ValueError
        return parts
    except ValueError:
        raise argparse.ArgumentTypeError("RGB value must be in the format R,G,B with three integers")


def main():
    parser = argparse.ArgumentParser(
        description="""Get images of segmentations obtained by the nnUNet.
        Images are 3d renderings of the CT, overlaid with the ground
        truth and predicted masks. Besides, includes a comparison
        of ground truth and predicted mask for an specified axial
        slice.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'path_to_cts',
        type=str,
        help="""Path to the directory containing the CT images in nifti
        format (.nii.gz)."""
    )
    parser.add_argument(
        'path_to_gts',
        type=str,
        help="""Path to the directory containing the ground truth
        segmentation masks in nifti format (.nii.gz)."""
    )
    parser.add_argument(
        'path_to_predictions',
        type=str,
        help="""Path to the directory containing the predicted
        segmentation masks in nifti format (.nii.gz)."""
    )
    parser.add_argument(
        'path_to_output',
        type=str,
        help="Path to the output directory."
    )
    parser.add_argument(
        '--path_to_json',
        type=str,
        default="nnunet_rendering.json",
        help="""Path to the JSON file containing custom parameters
        for each filename. The filename must match the filename of the
        ground truth nifti file. All attritutes of the Visualizer class
        are customizable. Besides, you can specify:
            'slice_idx' (int): axial slice of the CT with a comparison of
            the ground truth and the predicted mask,
            'fill_predicted_mask' (bool): set to True to fill predicted
            mask in the extracted axial slice (only countour by default),
            'fill_gt_mask' (bool): set to True to fill ground truth mask
            in the extracted axial slice (only countour by default)."""
    )
    parser.add_argument(
        '--gt_color',
        type=rgb_tuple,
        default='1,0,0',
        help="""Color for the ground truth mask expressed as 'R,G,B' with
        each value in the range [0,1]. Examples: blue-like (0.1,0.52,1.0)."""
    )
    parser.add_argument(
        '--prediction_color',
        type=rgb_tuple,
        default='1.0,0.84,0',
        help="""Color for the prediction mask expressed as 'R,G,B' with
        each value in the range [0,1]. Examples: magenta-like (0.83,0.07,0.35)."""
    )
    args = parser.parse_args()
    render_all(
        args.path_to_cts,
        args.path_to_gts,
        args.path_to_predictions,
        args.path_to_output,
        args.path_to_json,
        gt_color=args.gt_color,
        prediction_color=args.prediction_color
    )


if __name__ == "__main__":
    main()
