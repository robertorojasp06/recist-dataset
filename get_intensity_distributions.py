import numpy as np
import SimpleITK as sitk
import argparse
import pandas as pd
import concurrent.futures
from pathlib import Path


class Processor:
    def __init__(self):
        self.max_workers = 8
        self.min_hu = -1024
        self.max_hu = 3071
        self.bins_count = 100

    @property
    def bins(self):
        return np.linspace(self.min_hu, self.max_hu, self.bins_count)

    def _get_stats(self, path_to_image):
        print(f"filename: {Path(path_to_image).name}")
        image = sitk.ReadImage(path_to_image)
        array = sitk.GetArrayFromImage(image)
        stats = {
            "filename": Path(path_to_image).name,
            "slices": array.shape[0],
            "rows": array.shape[1],
            "columns": array.shape[2],
            "dtype": str(array.dtype),
            "mean": np.mean(array),
            "std": np.std(array),
            "min": np.min(array),
            "Q1": np.percentile(array, 25),
            "Q2": np.percentile(array, 50),
            "Q3": np.percentile(array, 75),
            "max": np.max(array)
        }
        return stats

    def _get_density(self, path_to_image):
        print(f"filename: {Path(path_to_image).name}")
        image = sitk.ReadImage(path_to_image)
        array = sitk.GetArrayFromImage(image)
        hist, _ = np.histogram(
            array,
            bins=self.bins,
            density=True
        )
        output = {
            "filename": Path(path_to_image).name,
            "density": hist,
            "bins": self.bins
        }
        return output

    def get_stats(self, path_to_images):
        paths_to_images = list(Path(path_to_images).glob('*.nii.gz'))        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(self._get_stats, paths_to_images))
        return results

    def get_densities(self, path_to_images):
        paths_to_images = list(Path(path_to_images).glob('*.nii.gz'))        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(self._get_density, paths_to_images))
        return results


def main():
    parser = argparse.ArgumentParser(
        description="""Get statistics and probability density
        distributions of CT images.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'path_to_images',
        type=str,
        help="""Path to the directory containing CT images in
        compressed nifti format."""
    )
    parser.add_argument(
        '--path_to_output',
        type=str,
        default=Path.cwd(),
        help="Path to the output directory."
    )
    parser.add_argument(
        '--max_threads',
        type=int,
        default=8,
        help="Maximum number of threads for multithreading."
    )
    args = parser.parse_args()
    processor = Processor()
    processor.max_workers = args.max_threads
    # Stats
    results = processor.get_stats(args.path_to_images)
    pd.DataFrame(results).to_csv(
        Path(args.path_to_output) / "stats.csv",
        index=False
    )
    # Densities
    results = processor.get_densities(args.path_to_images)
    results = {
        item["filename"]: {**{k: v for k, v in item.items() if k != "filename"}}
        for item in results
    }
    np.save(
        Path(args.path_to_output) / "densities.npy",
        results,
        allow_pickle=True
    )


if __name__ == "__main__":
    main()
