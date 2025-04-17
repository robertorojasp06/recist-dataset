import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import json
from pathlib import Path
from vedo import Volume, Plotter
from tqdm import tqdm


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
        self.mask_mesh_color_gt = 'red'
        self.mask_mesh_color_prediction = 'yellow'
        self.mask_mesh_alpha = 0.95
        self.mask_mesh_lighting = 'plastic'
        self.mask_mesh_smooth_niter = 15
        self.mask_mesh_smooth_boundary = True
        self.output_image_size =(512, 512)
        self.zoom_factor = 1.35
        self.screenshot_scale = 2
        self.background_color = 'black'
        self.interactive = False

    @property
    def path_to_output_vols(self):
        return Path(self.path_to_output) / "ct"

    @property
    def path_to_output_gt_overlaid(self):
        return Path(self.path_to_output) / "ct-gt-overlaid"

    @property
    def path_to_output_prediction_overlaid(self):
        return Path(self.path_to_output) / "ct-prediction-overlaid"

    def render(self, path_to_ct, path_to_gt_mask, path_to_prediction_mask):
        for path in (
            self.path_to_output_vols,
            self.path_to_output_gt_overlaid,
            self.path_to_output_prediction_overlaid
        ):
            Path(path).mkdir(exist_ok=True)
        # CT Volume as a mesh
        vol = Volume(path_to_ct)
        vol_mesh = vol.isosurface()
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
            "elevation": 90,
            "axes": 0,
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

    def render_all(self, path_to_cts, path_to_gts, path_to_predictions):
        paths_to_gts = sorted(list(Path(path_to_gts).glob('*.nii.gz')))
        paths_to_predictions = sorted(list(Path(path_to_predictions).glob('*.nii.gz')))
        paths_to_cts = sorted(
            [
                Path(path_to_cts) / f"{path.name.split('.nii.gz')[0]}_0000.nii.gz"
                for path in paths_to_gts
            ]
        )
        for path_to_ct, path_to_gt, path_to_prediction in tqdm(
            zip(
                paths_to_cts,
                paths_to_gts,
                paths_to_predictions
            ),
            total=len(paths_to_cts)
        ):
            tqdm.write(f"ct: {Path(path_to_ct).name}")
            self.render(
                str(path_to_ct),
                str(path_to_gt),
                str(path_to_prediction)
            )


def render_sample(path_to_ct, path_to_gt, path_to_prediction,
                  path_to_output, params):
    visualizer = Visualizer(path_to_output)
    for key, value in params.items():
        setattr(visualizer, key, value)
    visualizer.render(
        path_to_ct,
        path_to_gt,
        path_to_prediction
    )


def render_all(path_to_cts, path_to_gts, path_to_predictions,
               path_to_output, path_to_params):
    with open(path_to_params, 'r') as file:
        params_dict = json.load(file)
    paths_to_gts = sorted(list(Path(path_to_gts).glob('*.nii.gz')))
    paths_to_predictions = sorted(list(Path(path_to_predictions).glob('*.nii.gz')))
    paths_to_cts = sorted(
        [
            Path(path_to_cts) / f"{path.name.split('.nii.gz')[0]}_0000.nii.gz"
            for path in paths_to_gts
        ]
    )
    for path_to_ct, path_to_gt, path_to_prediction in tqdm(
        zip(
            paths_to_cts,
            paths_to_gts,
            paths_to_predictions
        ),
        total=len(paths_to_cts)
    ):
        tqdm.write(f"ct: {Path(path_to_ct).name}")
        render_sample(
            str(path_to_ct),
            str(path_to_gt),
            str(path_to_prediction),
            str(path_to_output)
        )


def test_sample():
    path_to_ct = "/media/cosmo/Data/fondef_ID23I10337/nnUNet/data/raw/Dataset536_HCUCH_FineTuningLung_draft/imagesTs/1.3.12.2.1107.5.1.4.83504.30000021071911510369800019660_0000.nii.gz"
    path_to_gt = "/media/cosmo/Data/fondef_ID23I10337/nnUNet/data/raw/Dataset536_HCUCH_FineTuningLung_draft/labelsTs/1.3.12.2.1107.5.1.4.83504.30000021071911510369800019660.nii.gz"
    path_to_prediction = "/media/cosmo/Data/fondef_ID23I10337/nnUNet/results/inference/Dataset536_HCUCH_FineTuningLung_draft/best_chk/3d_fullres_lr_0_0001/ensemble/1.3.12.2.1107.5.1.4.83504.30000021071911510369800019660.nii.gz"
    path_to_output = "/home/cosmo/nnunet-renders"
    visualizer = Visualizer(path_to_output)
    visualizer.render(
        path_to_ct,
        path_to_gt,
        path_to_prediction
    )


def test_all():
    # path_to_cts = "/media/cosmo/Data/fondef_ID23I10337/nnUNet/data/raw/Dataset536_HCUCH_FineTuningLung_draft/imagesTs"
    # path_to_gts = "/media/cosmo/Data/fondef_ID23I10337/nnUNet/data/raw/Dataset536_HCUCH_FineTuningLung_draft/labelsTs"
    # path_to_predictions = "/media/cosmo/Data/fondef_ID23I10337/nnUNet/results/inference/Dataset536_HCUCH_FineTuningLung_draft/best_chk/3d_fullres_lr_0_0001/ensemble"
    # path_to_output = "/home/cosmo/nnunet-renders"
    path_to_cts = "/media/robber/Nuevo vol/nnUNet/data/raw/Dataset536_HCUCH_FineTuningLung_draft/imagesTs"
    path_to_gts = "/media/robber/Nuevo vol/nnUNet/data/raw/Dataset536_HCUCH_FineTuningLung_draft/labelsTs"
    path_to_predictions = "/media/robber/Nuevo vol/nnUNet/results/inference/Dataset536_HCUCH_FineTuningLung_draft/best_chk/3d_fullres_lr_0_0001/ensemble"
    path_to_output = "/home/robber/nnunet-renders"
    visualizer = Visualizer(path_to_output)
    visualizer.render_all(
        path_to_cts,
        path_to_gts,
        path_to_predictions
    )


def main():
    test_all()


if __name__ == "__main__":
    main()
