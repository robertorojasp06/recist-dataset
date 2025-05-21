import pandas as pd
import argparse
import subprocess
import pydicom
import json
import concurrent.futures
from pathlib import Path


class Formatter:
    def __init__(self):
        self.max_workers = 4

    def find_selected_dicoms(self, path_to_dicom_folders,
                             path_to_series, path_to_patients):
        patients_df = pd.read_csv(path_to_patients)
        with open(path_to_series, 'r') as file:
            series_df = pd.DataFrame(json.load(file))
        series_df = series_df.merge(
            patients_df[["patient_id", "subset"]],
            on="patient_id",
            how="left"
        )
        series_df = series_df[["uuid", "subset"]].copy()
        series_df["uuid"] = series_df["uuid"].astype(str)
        selected_uuids = series_df["uuid"].unique().tolist()
        series_df.set_index('uuid', inplace=True)
        paths_to_folders = [
            str(path.parent)
            for path in Path(path_to_dicom_folders).rglob('*.dcm')
        ]
        paths_to_folders = list(set(paths_to_folders))
        selected_dicoms = [
            {
                "path_to_folder": path,
                "uuid": pydicom.dcmread(next(Path(path).glob('*.dcm'))).SeriesInstanceUID
            }
            for path in paths_to_folders
        ]
        selected_dicoms = [
            {
                **item,
                "subset": series_df.loc[item["uuid"], "subset"]
            }
            for item in selected_dicoms
            if item["uuid"] in selected_uuids
        ]
        return selected_dicoms

    def convert_dcm_to_nii(self, path_to_dicom, path_to_output_folder):
        Path(path_to_output_folder).mkdir(parents=True, exist_ok=True)
        series_uuid = pydicom.dcmread(
            list(Path(path_to_dicom).rglob('*.dcm'))[0],
            stop_before_pixels=True
        ).SeriesInstanceUID
        command = [
            "dcm2niix",
            "-z", "y",
            "-f", series_uuid,
            "-o", str(path_to_output_folder),
            "-b", "n",
            str(path_to_dicom)
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print("Error:", result.stderr)
            raise RuntimeError(f"dcm2niix conversion failed for {path_to_dicom}")
        print("Conversion successful:", result.stdout)

    def format_all(self, path_to_dicom_folders, path_to_output,
                   path_to_series, path_to_patients):
        selected_dicoms = self.find_selected_dicoms(
            path_to_dicom_folders,
            path_to_series,
            path_to_patients
        )
        input_tuples = [
            (item["path_to_folder"], Path(path_to_output) / item["subset"] / "images")
            for item in selected_dicoms
        ]
        paths_to_dicom, paths_to_output_folder = zip(*input_tuples)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            executor.map(
                self.convert_dcm_to_nii,
                paths_to_dicom,
                paths_to_output_folder
            )


def main():
    parser = argparse.ArgumentParser(
        description="""Convert DICOM series into NIfTI and following
        the format expected by the recist-dataset repository.
        DICOM-to-NIfTI conversion is based on dcm2niix
        utility."""
    )
    parser.add_argument(
        'path_to_dicom',
        type=str,
        help="""Path to the directory containing the folders containing
        the DICOM files. Recursive search is done to identify DICOM folders
        for each series."""
    )
    parser.add_argument(
        'path_to_output',
        type=str,
        help="Path to the directory to save output files."
    )
    parser.add_argument(
        'path_to_series',
        type=str,
        help="""Path to the 'series.json' file containing information
        about the series."""
    )
    parser.add_argument(
        'path_to_patients',
        type=str,
        help="""Path to the 'patients.csv' file containing information
        about patients."""
    )
    args = parser.parse_args()
    formatter = Formatter()
    formatter.format_all(
        args.path_to_dicom,
        args.path_to_output,
        args.path_to_series,
        args.path_to_patients
    )


if __name__ == "__main__":
    main()
