import argparse
import json
import pandas as pd
import argparse
import SimpleITK as sitk
import concurrent.futures
from pathlib import Path
from tqdm import tqdm


DROP_FROM_PATIENTS = [
    'patient_code',
    'first_study_uuid',
    'first_study_accession_number',
    'protocol'
]
DROP_FROM_SERIES = [
    'status',
    'study_time',
    'patient_code',
    'orthanc_uuid',
    'study_id'
]


def remove_keys(d, keys):
    new_d = dict(d)
    for key in keys:
        del new_d[key]
    return new_d


def get_final_files_df(path_to_train_ct, path_to_test_ct):
    final_files = []
    for subset, path_to_images in zip(
        ["train", "test"],
        [path_to_train_ct, path_to_test_ct]
    ):
        final_files.extend(
            [
                {
                    "filename": path.name,
                    "uuid": path.name.split('.nii.gz')[0],
                    "subset": subset
                }
                for path in Path(path_to_images).glob('*.nii.gz')
            ]
        )
    final_files_df = pd.DataFrame(final_files)
    assert len(final_files) == len(final_files_df["uuid"].unique().tolist()), "Some nifti files share the filaname."
    return final_files_df


def get_slice_spacings_helper(path):
    image = sitk.ReadImage(path)
    slice_spacing = {
        f"{path.name.split('.nii.gz')[0]}": float(image.GetMetaData("pixdim[3]"))
    }
    return slice_spacing


def get_slice_spacings(path_to_train_ct, path_to_test_ct):
    paths_to_nifti = (
        list(Path(path_to_train_ct).glob('*.nii.gz')) +
        list(Path(path_to_test_ct).glob('*.nii.gz'))
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        slice_spacings = list(tqdm(executor.map(get_slice_spacings_helper, paths_to_nifti), total=len(paths_to_nifti)))
    # slice_spacings = {
    #     list(item.keys())[0]: list(item.values())[0]
    #     for item in slice_spacings
    # }
    slice_spacings = {
        key: value
        for item in slice_spacings
        for key, value in item.items()
    }
    return slice_spacings


def map_name(name):
    result = name.split()[0]
    if result.lower() == 'portal':
        region = "abdomen"
    elif result.lower() == 'torax':
        region = "thorax"
    else:
        raise ValueError(f"unexpected name: {name}")
    return region


def process_recist_measurements(path_to_recist):
    df = pd.read_csv(path_to_recist)
    if 'name' in df.columns:
        df['region'] = df['name'].apply(map_name)
        df = df.drop(columns='name', errors='ignore')
        cols = df.columns.tolist()
        cols.insert(6, cols.pop(cols.index('region')))
        df = df[cols]
    return df


def main():
    parser = argparse.ArgumentParser(
        description="""Obtain final metadata files from raw metadata files.
        Only metadata of files included in the final dataset is kept.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'path_to_train_ct',
        type=str,
        help="""Path to the folder containing the nifti files of
        CT images in the training set."""
    )
    parser.add_argument(
        'path_to_test_ct',
        type=str,
        help="""Path to the folder containing the nifti files of
        CT images in the test set."""
    )
    parser.add_argument(
        'path_to_series',
        type=str,
        help="""Path to the series.json file containing the metadata
        for each series."""
    )
    parser.add_argument(
        'path_to_patients',
        type=str,
        help="""Path to the patients.csv file containing the information
        about each patient."""
    )
    parser.add_argument(
        'path_to_windows_mapping',
        type=str,
        help="""Path to the 'windows_mapping.json' file containing
        the mapping between each filename and the corresponding
        window for windowing normalization."""
    )
    parser.add_argument(
        '--path_to_recist',
        type=str,
        default=None,
        help="Path to the 'recist_measurements.csv' file"
    )
    parser.add_argument(
        '--path_to_output',
        type=str,
        default=Path.cwd(),
        help="Path to the output directory."
    )
    args = parser.parse_args()
    # Get list of final files
    final_files_df = get_final_files_df(
        args.path_to_train_ct,
        args.path_to_test_ct
    )
    final_files_uuids = final_files_df["uuid"].tolist()
    final_files_filenames = final_files_df["filename"].tolist()
    # Keep only final files in series.json
    with open(args.path_to_series, 'r') as file:
        raw_series = json.load(file)
    assert len(raw_series) == len(pd.DataFrame(raw_series)["uuid"].unique().tolist()), f"{Path(args.path_to_series).name} file has repeated 'uuid' identifiers."
    final_series = [
        item
        for item in raw_series
        if item['uuid'] in final_files_uuids
    ]
    final_series_df = pd.DataFrame(final_series)
    if len(final_files_df) > len(final_series):
        final_series_uuids = final_series_df['uuid'].unique().tolist()
        series_without_metadata = [
            row["filename"]
            for _, row in final_files_df.iterrows()
            if row["uuid"] not in final_series_uuids
        ]
        raise AssertionError(f"CT images without metadata in {Path(args.path_to_series).name}: {', '.join(series_without_metadata)}")
    # Keep only final files in windows_mapping.json
    with open(args.path_to_windows_mapping, 'r') as file:
        raw_windows_mapping = json.load(file)
    final_windows_mapping = {
        key: value
        for key, value in raw_windows_mapping.items()
        if key in final_files_filenames
    }
    if len(final_files_df) > len(final_windows_mapping):
        final_windows_filenames = list(final_windows_mapping.keys())
        series_without_window = [
            row["filename"]
            for _, row in final_files_df.iterrows()
            if row["filename"] not in final_windows_filenames
        ]
        raise AssertionError(f"CT images without specified window in {Path(args.path_to_windows_mapping).name}: {', '.join(series_without_window)}")
    # Keep only patients with images in the final files
    final_patients_ids = final_series_df["patient_id"].unique().tolist()
    raw_patients_df = pd.read_csv(args.path_to_patients)
    final_patients_df = raw_patients_df[raw_patients_df['patient_id'].isin(final_patients_ids)]
    # Remove unnecessary columns from patients
    final_patients_df = final_patients_df.drop(columns=DROP_FROM_PATIENTS, errors='ignore')
    # Remove unnecessary fields from series and add slice spacing
    slice_spacings = get_slice_spacings(
        args.path_to_train_ct,
        args.path_to_test_ct
    )
    final_series = [
        {
            **remove_keys(item, DROP_FROM_SERIES),
            "slice_spacing": slice_spacings.get(item["uuid"])
        }
        for item in final_series
    ]
    series_df = pd.DataFrame(final_series)
    if 'name' in series_df.columns:
        series_df['region'] = series_df['name'].apply(map_name)
        series_df = series_df.drop(columns='name', errors='ignore')
        cols = series_df.columns.tolist()
        cols.insert(1, cols.pop(cols.index('region')))
        series_df = series_df[cols]
        final_series = series_df.to_dict(orient='records')
    # Save final metadata
    print(f"Final images: {len(final_files_df)}")
    print(f"Final images with JSON metadata: {len(final_series_df)}")
    print(f"Final patients: {len(final_patients_df)}")
    with open(Path(args.path_to_output) / "series.json", 'w') as file:
        json.dump(
            final_series,
            file,
            indent=4
        )
    with open(Path(args.path_to_output) / "windows_mapping.json", 'w') as file:
        json.dump(
            final_windows_mapping,
            file,
            indent=4
        )
    final_patients_df.to_csv(
        Path(args.path_to_output) / "patients.csv",
        index=False
    )
    # Process RECIST measurements
    if args.path_to_recist:
        recist_df = process_recist_measurements(args.path_to_recist)
        recist_df.to_csv(
            Path(args.path_to_output) / Path(args.path_to_recist).name,
            index=False
        )


if __name__ == "__main__":
    main()
