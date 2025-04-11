import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import concurrent.futures
import json
import matplotlib.patheffects as path_effects
from pathlib import Path
from skimage.measure import find_contours
from skimage import io
from matplotlib_scalebar.scalebar import ScaleBar

from utils.plot import overlay_masks, show_box
from utils.mappings import LABEL_MAPPING


def get_label_description(performance, path_to_labels,
                          sp_en_mapping=LABEL_MAPPING):
    """Return english translated label.
    Return None if label value is not found in JSON file.
    Return spanish label if there is no mapping available."""
    with open(Path(path_to_labels) / f"{performance['study']}.json") as file:
        labels = json.load(file)
    label_description = labels.get(str(performance['foreground_label']), None)
    # Map to english version
    if label_description:
        label_description = sp_en_mapping.get(
            ','.join([item.strip() for item in label_description.split(',')]),
            label_description
        )
    return label_description


def get_performance_df(path_to_results, path_to_series, path_to_labels,
                       path_to_patients):
        with open(Path(path_to_results) / "performance.json", 'r') as file:
            performances = json.load(file)["bboxes"]
        with open(path_to_series, 'r') as file:
            series_df = pd.DataFrame(json.load(file))
        patients_df = pd.read_csv(path_to_patients)
        performances = [
            {
                "uuid": performance["study"],
                "patient_id": series_df.loc[series_df["uuid"] == performance['study'], "patient_id"].item(),
                "diagnosis": patients_df.loc[patients_df["patient_id"] == series_df.loc[series_df["uuid"] == performance['study'], "patient_id"].item(), "diagnosis"].item(),
                "slice_idx": performance["slice_idx"],
                "foreground_label": performance["foreground_label"],
                "mask_filename": f"{Path(performance['path_to_bbox_original']).stem}.png",
                "pixel_size": series_df.loc[series_df["uuid"] == performance['study'], "row_spacing"].item(),
                "label_description": get_label_description(performance, path_to_labels),
                "bbox_col_min_x0": performance["bbox_original"][0],
                "bbox_row_min_y0": performance["bbox_original"][1],
                "bbox_col_max_x1": performance["bbox_original"][2],
                "bbox_row_max_y1": performance["bbox_original"][3],
                "annotated_pixels": performance["annotated_pixels"],
                "predicted_pixels": performance["predicted_pixels"],
                "dice_score": performance["dice_score"]
            }
            for performance in performances
        ]
        return pd.DataFrame(performances)


class ROIExtractor:
    def __init__(self, path_to_output):
        self.path_to_output = path_to_output
        self.roi_shape = (140, 140)
        self.roi_linewidth = 3
        self.roi_edgecolor = (0, 0 ,1)
        self.fill_mask = False
        self.masks_alpha = 1.0
        self.predicted_color = (1, 0.4, 0.4)
        self.gt_color = (0, 1.0, 0)
        self.figsize = (4,4)
        self.scalebar_fontsize = 30
        self.add_text = True
        self.text_position = (0.6, 0.85) # (0, 0) is bottom left, (1, 1) is top right
        self.text_fontsize = 42
        self.text_fontweight = 'bold'
        self.output_format = 'png'
        self.dpi = 300
        self.max_workers = 4
        self.verbose = False
        self.test = False

    def plot_lesion(self, path_to_slice, path_to_predicted_mask,
                    path_to_gt_mask, bbox, pixel_size_mm, dice_score,
                    label_description):
        print(f"mask filename: {Path(path_to_predicted_mask.name)}")
        slice_ct = io.imread(path_to_slice)
        predicted_mask = io.imread(path_to_predicted_mask)
        gt_mask = io.imread(path_to_gt_mask)
        # Get contour masks
        if not self.fill_mask:
            predicted_contours = find_contours(predicted_mask > 0)
            gt_contours = find_contours(gt_mask > 0)
            predicted_mask = np.zeros_like(predicted_mask)
            gt_mask = np.zeros_like(gt_mask)
            for contour in gt_contours:
                gt_mask[contour[:, 0].astype('int'), contour[:, 1].astype('int')] = True
            for contour in predicted_contours:
                predicted_mask[contour[:, 0].astype('int'), contour[:, 1].astype('int')] = True            
        # Plot slice with overlaid mask
        overlay = overlay_masks(
            slice_ct,
            predicted_mask,
            gt_mask,
            predicted_color=self.predicted_color,
            gt_color=self.gt_color,
            alpha=self.masks_alpha
        )
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.imshow(overlay)
        show_box(
            bbox,
            ax,
            edgecolor=self.roi_edgecolor,
            linewidth=self.roi_linewidth
        )
        ax.set_axis_off()
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        path_to_output_slice = Path(self.path_to_output) / "slices"
        path_to_output_slice.mkdir(exist_ok=True)
        fig.savefig(
            path_to_output_slice / f"{Path(path_to_predicted_mask).stem}_{label_description.split(',')[0]}_{label_description.split(',')[1]}.{self.output_format}",
            transparent=True,
            dpi=self.dpi
        )
        plt.close(fig)
        # Plot ROI centered lesion
        aug_shape = (
            slice_ct.shape[0] + 2 * self.roi_shape[0],
            slice_ct.shape[1] + 2 * self.roi_shape[1],
            3
        )
        aug_overlay = np.zeros(aug_shape)
        aug_overlay[
            self.roi_shape[0]:self.roi_shape[0] + slice_ct.shape[0],
            self.roi_shape[1]:self.roi_shape[1] + slice_ct.shape[1],
            ...
        ] = overlay
        # Compute offsets for bounding box corners to satisfy the ROI size
        delta_row = int((self.roi_shape[0] - (bbox[3] - bbox[1])) // 2)
        delta_col = int((self.roi_shape[1] - (bbox[2] - bbox[0])) // 2)
        aug_row_min = bbox[1] + self.roi_shape[0] - delta_row
        aug_row_max = aug_row_min + self.roi_shape[0]
        aug_col_min = bbox[0] + self.roi_shape[1] - delta_col
        aug_col_max = aug_col_min + self.roi_shape[1]
        output_overlay = aug_overlay[aug_row_min:aug_row_max, aug_col_min:aug_col_max]
        new_bbox = [
            bbox[0] - (aug_col_min - self.roi_shape[1]),
            bbox[1] - (aug_row_min - self.roi_shape[0]),
            bbox[2] - (aug_col_min - self.roi_shape[1]),
            bbox[3] - (aug_row_min - self.roi_shape[0])
        ]
        # Plot ROI centered lesion with overlaid masks
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.imshow(output_overlay)
        show_box(
            new_bbox,
            ax,
            scaling=1.025,
            edgecolor=self.roi_edgecolor,
            linewidth=self.roi_linewidth
        )
        # Draw scalebar
        scalebar = ScaleBar(
            pixel_size_mm,
            units="mm",
            fixed_value=25,
            location='lower center',
            box_alpha=1,
            box_color='black',
            font_properties={"size": self.scalebar_fontsize},
            color='white'
        )
        ax.add_artist(scalebar)
        # Draw text with dice score
        if self.add_text:
            text = ax.text(
                self.text_position[0],
                self.text_position[1],
                f"DSC: {round(dice_score, ndigits=2)}",
                color=self.predicted_color,
                fontsize=self.text_fontsize,
                fontweight=self.text_fontweight,
                transform=ax.transAxes
            )
            text.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground='black')])
        # Save figure
        ax.set_axis_off()
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        path_to_output_roi = Path(self.path_to_output) / "rois"
        path_to_output_roi.mkdir(exist_ok=True)
        fig.savefig(
            path_to_output_roi / f"{Path(path_to_predicted_mask).stem}_{label_description.split(',')[0]}_{label_description.split(',')[1]}.{self.output_format}",
            transparent=True,
            dpi=self.dpi
        )
        plt.close(fig)

    def plot_lesions(self, path_to_results, path_to_series, performance_df):
        with open(Path(path_to_results) / "performance.json", 'r') as file:
            performances = json.load(file)["bboxes"]
        with open(path_to_series, 'r') as file:
            series_df = pd.DataFrame(json.load(file))
        paths_to_slices = []
        paths_to_predicted_masks = []
        paths_to_gt_masks = []
        bboxes = []
        pixel_sizes = []
        dice_scores = []
        label_descriptions = []
        if self.test:
            performances = [
                item
                for item in performances[:100]
            ]
        for performance in performances:
            paths_to_slices.append(Path(path_to_results) / "original_size" / f"{performance['study']}_slice{performance['slice_idx']}.png")
            paths_to_predicted_masks.append(Path(path_to_results) / "output_masks" / f"{Path(performance['path_to_bbox_original']).stem}.png")
            paths_to_gt_masks.append(Path(path_to_results) / "gt_masks" / f"{Path(performance['path_to_bbox_original']).stem}.png")
            bboxes.append(performance["bbox_original"])
            pixel_sizes.append(series_df.loc[series_df["uuid"] == performance['study'], "row_spacing"].item())
            dice_scores.append(performance["dice_score"])
            label_descriptions.append(performance_df.loc[
                (performance_df["uuid"] == performance['study']) &
                (performance_df["bbox_col_min_x0"] == performance["bbox_original"][0]) &
                (performance_df["bbox_row_min_y0"] == performance["bbox_original"][1]) &
                (performance_df["bbox_col_max_x1"] == performance["bbox_original"][2]) &
                (performance_df["bbox_row_max_y1"] == performance["bbox_original"][3]) &
                (performance_df["slice_idx"] == performance["slice_idx"]),
                "label_description"
            ].item())
        # Extract ROIs
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            executor.map(
                self.plot_lesion,
                paths_to_slices,
                paths_to_predicted_masks,
                paths_to_gt_masks,
                bboxes,
                pixel_sizes,
                dice_scores,
                label_descriptions
            )


def main():
    parser = argparse.ArgumentParser(
         description= """TODO""",
         formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'path_to_results',
        type=str,
        help="""Path to the directory containing the outputs obtained
        from the 'evaluate_CT_dataset.py' script. It must have
        the 'performance.json' file and the following
        subbirectories:'original_size', 'output_masks',
        'gt_masks'."""
    )
    parser.add_argument(
        'path_to_labels',
        type=str,
        help="""Path to the folder containing the JSON files with the
        mapping between voxel values and lesion labels for test CT masks.
        Make sure to use the labels corresponding to the CT masks
        evaluated in the 'evaluate_CT_dataset.py'."""
    )
    parser.add_argument(
		'path_to_output',
		type=str,
		help="Path to the output directory."
	)
    parser.add_argument(
        '--path_to_series',
        type=str,
        default=Path.cwd() / "data" / "metadata" / "series.json",
        help="""Path to the 'series.json' file."""
    )
    parser.add_argument(
        '--roi_shape',
        nargs=2,
        type=int,
        default=[140, 140],
        help="ROI shape expressed as ROWS COLUMNS."
    )
    parser.add_argument(
        '--output_format',
        type=str,
        choices=['png', 'svg'],
        default='png',
        help="Format of output images."
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help="Dots per inch for output raster images."
    )
    parser.add_argument(
        '--predicted_color',
        nargs=3,
        type=float,
        default=[1, 0.4, 0.4],
        help="Color for predicted mask expressed as R G B."
    )
    parser.add_argument(
        '--gt_color',
        nargs=3,
        type=float,
        default=[0, 1.0, 0],
        help="Color for ground truth mask expressed as R G B."
    )
    parser.add_argument(
        '--bbox_color',
        nargs=3,
        type=float,
        default=[0, 0, 1.0],
        help="Color for bounding box expressed as R G B."
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help="""Add this flag to run the script on the first
        100 bounding boxes (ROIs)."""
    )
    parser.add_argument(
        '--fill_mask',
        action='store_true',
        help="""Add this flag to display filled masks.
        Only contours are displayed by default."""
    )
    parser.add_argument(
        '--add_dice',
        action='store_true',
        help="""Add this flag to add the dice score."""
    )
    args = parser.parse_args()
    path_to_output_slices = Path(args.path_to_output) / ("png")
    #path_to_output_imgs.mkdir(parents=True, exist_ok=True)


def test_concurrent():
    path_to_results = "/media/robber/Nuevo vol/MedSAM/results/MedSAM-HITL-iteration-3/MedSAM-ViT-B-20241229-0600"
    path_to_series = Path.cwd() / "data" / "metadata" / "series.json"
    path_to_output = "/home/robber/outputs"
    path_to_labels = "/home/robber/fondef-ID23I10337/resultados/medsam-finetuning/MedSAM-HITL-iteration-3/data/original/test/nifti/labels"
    path_to_patients = Path.cwd() / "data" / "metadata" / "patients.csv"
    # Create performance csv
    performance_df = get_performance_df(
        path_to_results,
        path_to_series,
        path_to_labels,
        path_to_patients
    )
    # Plot lesions with the ROIExtractor
    extractor = ROIExtractor(path_to_output)
    extractor.output_format = 'svg'
    extractor.fill_mask = False
    extractor.predicted_color = (230/255, 97/255, 0/255)
    extractor.gt_color = (255/255, 194/255, 10/255)
    extractor.roi_edgecolor = (64/255, 176/255, 166/255)
    extractor.masks_alpha = 1.0
    extractor.roi_shape = (140, 140)
    extractor.figsize = (4,4)
    extractor.scalebar_fontsize = 30
    extractor.text_fontsize = 42
    extractor.max_workers = 8
    extractor.test = False
    extractor.add_text = True
    # new
    extractor.text_position = (0.05, 0.9)
    extractor.text_fontsize = 30
    extractor.text_fontweight = "normal"
    extractor.plot_lesions(
        path_to_results,
        path_to_series,
        performance_df
    )
    with open(Path(path_to_output) / "parameters.json", 'w') as file:
        json.dump(
            vars(extractor),
            file,
            indent=4
        )


def test_csv():
    path_to_results = "/media/robber/Nuevo vol/MedSAM/results/MedSAM-HITL-iteration-3/MedSAM-ViT-B-20241229-0600"
    path_to_series = Path.cwd() / "data" / "metadata" / "series.json"
    path_to_labels = "/home/robber/fondef-ID23I10337/resultados/medsam-finetuning/MedSAM-HITL-iteration-3/data/original/test/nifti/labels"
    path_to_output = "/home/robber/outputs"
    path_to_patients = Path.cwd() / "data" / "metadata" / "patients.csv"
    performance_df = get_performance_df(
        path_to_results,
        path_to_series,
        path_to_labels,
        path_to_patients
    )
    performance_df.to_csv(
        Path(path_to_output) / "bbox_performances.csv",
        index=False
    )


if __name__ == "__main__":
    test_concurrent()
