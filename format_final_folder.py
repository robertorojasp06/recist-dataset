import argparse
import shutil
import pandas as pd
import json
import concurrent.futures
from pathlib import Path
from tqdm import tqdm


class Formatter:
    def __init__(self, path_to_final_folder, path_to_output):
        self.path_to_final_folder = path_to_final_folder
        self.path_to_output = path_to_output
        self.train_alias = "train"
        self.test_alias = "test"
        self.thorax_alias = "thorax"
        self.abdomen_alias = "abdomen"

    def _get_series(self):
        # Add 'subset' to series_metadata
        with open(Path(self.path_to_final_folder) / "metadata" / "series.json", 'r') as file:
            series = json.load(file)
        paths_to_cts = [
            *list((Path(self.path_to_final_folder) / "images" / "train" / "images").glob('*.nii.gz')),
            *list((Path(self.path_to_final_folder) / "images" / "test" / "images").glob('*.nii.gz'))
        ]
        subset_mapping = {
            path.name: path.parts[-3]
            for path in paths_to_cts
        }
        series = [
            {
                **item,
                'subset': subset_mapping.get(f"{item['uuid']}.nii.gz")
            }
            for item in series
        ]
        return series

    def _save_datapoint(self, series_metadata):
        # Construct path to output image
        path_to_src_image = Path(self.path_to_final_folder) / "images" / series_metadata['subset'] / "images" / f"{series_metadata['uuid']}.nii.gz"
        path_to_dst_image = Path(self.path_to_output) / "images" / series_metadata['region'] / series_metadata['subset'] / "images" / f"{series_metadata['patient_id']:03d}_{series_metadata['uuid']}.nii.gz"
        path_to_dst_image.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy2(path_to_src_image, path_to_dst_image)
        # Construct path to output mask
        path_to_src_mask = Path(self.path_to_final_folder) / "images" / series_metadata['subset'] / "masks" / f"{series_metadata['uuid']}.nii.gz"
        path_to_dst_mask = Path(self.path_to_output) / "images" / series_metadata['region'] / series_metadata['subset'] / "masks" / f"{series_metadata['patient_id']:03d}_{series_metadata['uuid']}.nii.gz"
        path_to_dst_mask.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy2(path_to_src_mask, path_to_dst_mask)
        # Construct path to output label
        path_to_src_labels = Path(self.path_to_final_folder) / "images" / series_metadata['subset'] / "labels" / f"{series_metadata['uuid']}.json"
        path_to_dst_labels = Path(self.path_to_output) / "images" / series_metadata['region'] / series_metadata['subset'] / "labels" / f"{series_metadata['patient_id']:03d}_{series_metadata['uuid']}.json"
        path_to_dst_labels.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy2(path_to_src_labels, path_to_dst_labels)

    def format_image_data(self):
        series = self._get_series()
        outputs = [None] * len(series)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            tasks = {
                executor.submit(
                    self._save_datapoint,
                    series_meta
                ): i for i, series_meta in enumerate(series)
            }
            for task in tqdm(concurrent.futures.as_completed(tasks), total=len(series)):
                i = tasks[task]
                try:
                    outputs[i] = task.result()
                except Exception as e:
                    print(f"Task {i} failed: {e}")
                    raise

    def format_metadata(self):
        # Copy patients and series
        path_to_src_series = Path(self.path_to_final_folder) / "metadata" / "series.json"
        path_to_dst_series = Path(self.path_to_output) / "metadata" / "series.json"
        path_to_dst_series.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy2(path_to_src_series, path_to_dst_series)
        path_to_src_patients = Path(self.path_to_final_folder) / "metadata" / "patients.csv"
        path_to_dst_patients = Path(self.path_to_output) / "metadata" / "patients.csv"
        shutil.copy2(path_to_src_patients, path_to_dst_patients)
        # Update filename in windows mapping
        with open(Path(self.path_to_final_folder) / "metadata" / "windows_mapping.json", 'r') as file:
            windows_mapping = json.load(file)
        series = self._get_series()
        windows_mapping = {
            f"{meta['patient_id']:03d}_{meta['uuid']}.nii.gz": windows_mapping[f"{meta['uuid']}.nii.gz"]
            for meta in series
        }
        with open(Path(self.path_to_output) / "metadata" / "windows_mapping.json", 'w') as file:
            json.dump(windows_mapping, file, indent=4)
        # Update recist measurements
        recist_df = pd.read_csv(Path(self.path_to_final_folder) / "recist_measurements.csv")
        recist_df['filename'] = recist_df.apply(lambda row: f"{row['patient_id']:03d}_{row['filename']}", axis=1)
        recist_df.to_csv(
            Path(self.path_to_output) / "recist_measurements.csv",
            index=False
        )


def main():
    parser = argparse.ArgumentParser(
        description="""Give final folder the format suggested
        after first round of revision. Data is now grouped into
        thorax and abdomen folders, with train/test subfolders.
        Also, filenames now have a prefix indicating
        the patient_id, e.g. patient_001_1.3.12.2.1107.5.1.4.83504.30000021102112360685900001538.nii.gz.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'path_to_final_folder',
        type=str,
        help="""Path to the final folder."""
    )
    parser.add_argument(
        'path_to_output',
        type=str,
        help="Path to the output folder."
    )
    args = parser.parse_args()
    formatter = Formatter(
        args.path_to_final_folder,
        args.path_to_output
    )
    formatter.format_image_data()
    formatter.format_metadata()


if __name__ == "__main__":
    main()
