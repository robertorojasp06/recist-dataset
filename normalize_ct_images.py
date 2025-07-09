import os
import SimpleITK as sitk
import argparse
import concurrent.futures
from pathlib import Path

from utils.windowing import (
    WINDOWS,
    get_windows_mapping,
    check_windows_mapping,
    normalize_ct
)


class ParallelProcessor:
	def __init__(self, path_to_output, windows=WINDOWS):
		self.path_to_output = path_to_output
		self.windows = windows

	def normalize_ct(self, path_to_ct, window_params):
		print(f"processing: {Path(path_to_ct).name}")
		ct_image = sitk.ReadImage(path_to_ct)
		ct_array = sitk.GetArrayFromImage(ct_image)
		ct_array_norm = normalize_ct(ct_array, window_params)
		ct_image_norm = sitk.GetImageFromArray(ct_array_norm)
		ct_image_norm.CopyInformation(ct_image)
		sitk.WriteImage(
			ct_image_norm,
			Path(self.path_to_output) / path_to_ct.name
		)


def main():
	parser = argparse.ArgumentParser(
		description="Normalize CT images using the Windowing approach.",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)
	parser.add_argument(
		'path_to_cts',
		type=str,
		help = """Path to to the directory containing the CT images saved as
		compressed nifti files (.nii.gz)."""
	)
	parser.add_argument(
		'path_to_output',
		type=str,
		help="Path to the directory to save the output files."
	)
	parser.add_argument(
		'window',
		type=str,
		help=f"""Window for CT normalization: {list(WINDOWS.keys())}.
		This window is applied on all CTs. Alternatively, you can provide
		the path to a JSON file with a dictionary containing the
		mapping between filenames and windows."""
	)
	parser.add_argument(
		'--max_workers',
		type=int,
		default=4,
		help=f"""Max number of processes running in parallel. Don't
		use a value higher than your CPU cores: {os.cpu_count()}."""
	)
	args = parser.parse_args()
	windows_mapping = get_windows_mapping(
		args.window,
		args.path_to_cts
	)
	check_windows_mapping(
		windows_mapping,
		args.path_to_cts
	)
	paths_to_cts = list(Path(args.path_to_cts).glob('*.nii.gz'))
	window_values = [
		WINDOWS.get(windows_mapping.get(path.name))
		for path in paths_to_cts
	]
	processor = ParallelProcessor(args.path_to_output)
	with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as pool:
		pool.map(
			processor.normalize_ct,
			paths_to_cts,
			window_values
		)


if __name__ == "__main__":
	main()
