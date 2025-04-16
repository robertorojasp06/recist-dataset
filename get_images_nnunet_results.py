import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path
from vedo import Volume, Plotter, show
from PIL import Image

from utils.plot import overlay_segmentation
from utils.windowing import WINDOWS


def visualize_all_normalized(path_to_ct, path_to_mask):
    # CT Volume
    vol = Volume(path_to_ct)
    vol_mesh = vol.isosurface()
    vol_mesh.c("ivory").alpha(0.25).lighting("glossy")  # appearance settings
    # Shrink the bounding box a bit
    xmin, xmax, ymin, ymax, zmin, zmax = vol_mesh.bounds()
    vol_mesh.cut_with_box([
        xmin + 20, xmax - 20,
        ymin + 70, ymax - 70,
        zmin + 20, zmax - 20
    ])
    vol_mesh = vol_mesh.extract_largest_region()
    # Mask Volume
    mask_mesh = Volume(path_to_mask).isosurface(0.5)
    mask_mesh.c("red").alpha(0.95).lighting('plastic')
    mask_mesh = mask_mesh.smooth(niter=15, boundary=True)

    # Show the volume
    plotter = Plotter(bg="black")
    plotter.show(
        vol_mesh,
        mask_mesh,
        elevation=90,
        axes=0,
        zoom=True,
        size=[512, 512]
    )
    plotter.zoom(1.35)
    plotter.screenshot('output_image.png', scale=2)


class Projector:
    def __init__(self, path_to_output):
        self.path_to_output = path_to_output
        self.figsize = (8, 8)
        self.dpi = 300
        self.epsilon = 1e-6

    @property
    def path_to_ct_mip(self):
        return Path(self.path_to_output) / "mip_ct"

    @property
    def path_to_overlaid_mip(self):
        return Path(self.path_to_output) / "mip_overlaid"

    @property
    def path_to_overlaid_masked_mip(self):
        return Path(self.path_to_output) / "mip_overlaid_masked"

    def _get_masked_overlay(self, ct_array, gt_array, mask_alpha=0.8,
                            gray_alpha=0.5, window=None):
        if window:
            lower_bound = window["L"] - window["W"] / 2
            upper_bound = window["L"] + window["W"] / 2
            clipped_ct_array = np.where(
                (ct_array >= lower_bound) & (ct_array <= upper_bound),
                ct_array,
                0
            )
            ct_mip = np.max(
                # np.clip(ct_array, lower_bound, upper_bound),
                clipped_ct_array,
                axis=1
            )
            plt.imshow(clipped_ct_array[100])
            plt.show()
        else:
            ct_mip = np.max(ct_array, axis=1)
        gt_mip = np.max(gt_array, axis=1)
        masked_ct_mip = np.max(
            np.where(gt_array, ct_array, 0),
            axis=1
        )
        masked_ct_mip_3c = np.stack(3 * [np.zeros_like(ct_mip)], axis=-1)
        masked_ct_mip_3c[..., 1] = (masked_ct_mip - np.min(masked_ct_mip)) / (np.max(masked_ct_mip) - np.min(masked_ct_mip) + self.epsilon) # green
        ct_mip_3c = np.stack(
            3 * [(ct_mip - np.min(ct_mip)) / (np.max(ct_mip) - np.min(ct_mip) + self.epsilon)],
            axis=-1
        )
        overlaid = np.where(
            gt_mip[..., None] != 0,
            (1 - mask_alpha ) * ct_mip_3c + mask_alpha * masked_ct_mip_3c,
            gray_alpha * ct_mip_3c
        )
        return overlaid

    def _project_volumes(self, path_to_ct, path_to_gt,
                         path_to_prediction):
        for path in (
            self.path_to_ct_mip,
            self.path_to_overlaid_mip,
            self.path_to_overlaid_masked_mip
        ):
            Path(path).mkdir(exist_ok=True)
        # Read images
        ct_image = sitk.ReadImage(path_to_ct)
        ct_array = sitk.GetArrayFromImage(ct_image)
        gt_array = sitk.GetArrayFromImage(sitk.ReadImage(path_to_gt))
        prediction_array = sitk.GetArrayFromImage(sitk.ReadImage(path_to_prediction))
        # Get masked CT
        masked_ct_array = np.where(
            gt_array,
            ct_array,
            0
        )
        # Get MIP projections
        ct_mip = np.max(ct_array, axis=1)
        masked_ct_mip = np.max(masked_ct_array, axis=1)
        gt_mip = np.max(gt_array, axis=1)
        prediction_mip = np.max(prediction_array, axis=1)
        # print(f"gt foreground label: {[item for item in np.unique(gt_mip) if item != 0]}")
        # print(f"prediction foreground label: {[item for item in np.unique(prediction_mip) if item != 0]}")
        # Save MIP: CT
        plt.figure(figsize=(8, 8))
        plt.imshow(ct_mip, cmap='gray')
        plt.axis('off')
        plt.savefig(
            Path(self.path_to_ct_mip) / f"{Path(path_to_ct).name.split('.nii.gz')[0]}.png",
            transparent=True,
            bbox_inches='tight',
            dpi=self.dpi
        )
        plt.close()
        # Save MIP: overlaid mask
        overlaid_mip = overlay_segmentation(
            ct_mip,
            prediction_mip,
            alpha=0.25
        )
        plt.figure(figsize=(8, 8))
        plt.imshow(overlaid_mip)
        plt.axis('off')
        plt.savefig(
            Path(self.path_to_overlaid_mip) / f"{Path(path_to_ct).name.split('.nii.gz')[0]}.png",
            transparent=True,
            bbox_inches='tight',
            dpi=self.dpi
        )
        plt.close()

        visualize_all(path_to_ct, path_to_gt)


def test_projection():
    path_to_ct = "/media/robber/Nuevo vol/recist-dataset/data/images/final/test-updated-beta/images/1.3.12.2.1107.5.1.4.83504.30000021071911510369800019660.nii.gz"
    # path_to_ct = "/media/robber/Nuevo vol/nnUNet/data/raw/Dataset536_HCUCH_FineTuningLung_draft/imagesTs/1.3.12.2.1107.5.1.4.83504.30000021071911510369800019660_0000.nii.gz"
    path_to_gt = "/media/robber/Nuevo vol/nnUNet/data/raw/Dataset536_HCUCH_FineTuningLung_draft/labelsTs/1.3.12.2.1107.5.1.4.83504.30000021071911510369800019660.nii.gz"
    path_to_prediction = "/media/robber/Nuevo vol/nnUNet/results/inference/Dataset536_HCUCH_FineTuningLung_draft/best_chk/3d_fullres_lr_0_0001/ensemble/1.3.12.2.1107.5.1.4.83504.30000021071911510369800019660.nii.gz"
    path_to_output = "/home/robber/mip_projections"
    # projector = Projector(path_to_output)
    # projector._project_volumes(
    #     path_to_ct,
    #     path_to_gt,
    #     path_to_prediction
    # )
    visualize_all(
        path_to_ct,
        path_to_gt
    )


def test_projection_normalized():
    path_to_ct = "/media/robber/Nuevo vol/nnUNet/data/raw/Dataset536_HCUCH_FineTuningLung_draft/imagesTs/1.3.12.2.1107.5.1.4.83504.30000021071911510369800019660_0000.nii.gz"
    path_to_gt = "/media/robber/Nuevo vol/nnUNet/data/raw/Dataset536_HCUCH_FineTuningLung_draft/labelsTs/1.3.12.2.1107.5.1.4.83504.30000021071911510369800019660.nii.gz"
    path_to_prediction = "/media/robber/Nuevo vol/nnUNet/results/inference/Dataset536_HCUCH_FineTuningLung_draft/best_chk/3d_fullres_lr_0_0001/ensemble/1.3.12.2.1107.5.1.4.83504.30000021071911510369800019660.nii.gz"
    path_to_output = "/home/robber/mip_projections"
    # projector = Projector(path_to_output)
    # projector._project_volumes(
    #     path_to_ct,
    #     path_to_gt,
    #     path_to_prediction
    # )
    visualize_all_normalized(
        path_to_ct,
        path_to_gt
    )


def main():
    test_projection_normalized()


if __name__ == "__main__":
    main()
