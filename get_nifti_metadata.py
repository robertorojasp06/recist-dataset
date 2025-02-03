import SimpleITK as sitk
import argparse
import concurrent.futures
import pandas as pd
import json
from pathlib import Path
 

class Extractor:
    def __init__(self):
        self.max_workers = 4

    def _process_image(self, path_to_image):
        print(f"filename: {Path(path_to_image).name}")
        image = sitk.ReadImage(path_to_image)
        metadata = {
            key: image.GetMetaData(key)
            for key in image.GetMetaDataKeys()
        }
        metadata = {
            "filename": Path(path_to_image).name,
            **metadata
        }
        resolution = {
            "filename": metadata["filename"],
            "rows": int(metadata["dim[2]"]),
            "columns": int(metadata["dim[1]"]),
            "slices": int(metadata["dim[3]"]),
            "row_spacing_mm": float(metadata["pixdim[2]"]),
            "column_spacing_mm": float(metadata["pixdim[1]"]),
            "slice_spacing_mm": float(metadata["pixdim[3]"])
        }
        output = {
            "metadata": metadata,
            "resolution": resolution
        }
        return output

    def process_images(self, path_to_data):
        paths_to_data = list(Path(path_to_data).glob('*.nii.gz'))
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = executor.map(self._process_image, paths_to_data)
        return results


def main():
    parser = argparse.ArgumentParser(
        description="""Get metadata from nifti files. Also,
        return a csv file with resolution information (volume size
        and voxel spacing).""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'path_to_data',
        type=str,
        help="Path to the directory containing the nifti files."
    )
    parser.add_argument(
        '--path_to_output',
        type=str,
        default=Path.cwd(),
        help="Path to the output directory"
    )
    parser.add_argument(
        '--max_threads',
        type=int,
        default=8,
        help="Maximum threads for multithreading."
    )
    args = parser.parse_args()
    extractor = Extractor()
    extractor.max_workers = args.max_threads
    results = list(extractor.process_images(args.path_to_data))
    metadata = [item["metadata"] for item in results]
    resolution = [item["resolution"] for item in results]
    with open("metadata.json", 'w') as file:
        json.dump(metadata, file, indent=4)
    pd.DataFrame(resolution).to_csv(
        "resolution.csv",
        index=False
    )


if __name__ == "__main__":
    main()
