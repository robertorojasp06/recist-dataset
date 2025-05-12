import os
import SimpleITK as sitk
import numpy as np
import argparse
import json
import concurrent.futures
from pathlib import Path
from typing import Optional, Union, Dict


WINDOWS = {
	"lung": {"L": -500, "W": 1400},
	"abdomen": {"L": 40, "W": 350},
	"bone": {"L": 400, "W": 1000},
	"air": {"L": -426, "W": 1000},
	"brain": {"L": 50, "W": 100},
	"mediastinum": {"L": 50, "W": 350}
}


def get_windows_mapping(window_arg: str, path_to_cts: str):
	if window_arg not in WINDOWS:
		with open(window_arg, 'r') as file:
			mapping = json.load(file)
	else:
		mapping = {
			path.name: window_arg
			for path in Path(path_to_cts).glob('*.nii.gz')
		}
	return mapping


def check_windows_mapping(mapping: Dict[str, str], path_to_cts: str):
	# Check wrong windows
	wrong_windows = [
		f"filename '{filename}' with wrong window '{window}'."
		for filename, window in mapping.items()
		if window not in WINDOWS
	]
	if wrong_windows:
		raise ValueError('\n'.join(wrong_windows))
	# Check all CTs have their corresponding window
	unassigned_cts = [
		f"filename '{path.name}' does not have a window assigned."
		for path in Path(path_to_cts).glob('*.nii.gz')
		if path.name not in mapping.keys()
	]
	if unassigned_cts:
		raise ValueError('\n'.join(unassigned_cts))


def normalize_ct(
    ct_array: np.ndarray,
    window: Optional[Dict[str, Union[int, float]]] = None,
    epsilon: float = 1e-6
) -> np.ndarray:
    if window:
        lower_bound = window["L"] - window["W"] / 2
        upper_bound = window["L"] + window["W"] / 2
        ct_array_pre = np.clip(ct_array, lower_bound, upper_bound)
        ct_array_pre = (
            (ct_array_pre - np.min(ct_array_pre) + epsilon)
            / (np.max(ct_array_pre) - np.min(ct_array_pre) + epsilon)
            * 255.0
        )
    else:
        lower_bound= np.percentile(ct_array[ct_array > 0], 0.5)
        upper_bound = np.percentile(ct_array[ct_array > 0], 99.5)
        ct_array_pre = np.clip(ct_array, lower_bound, upper_bound)
        ct_array_pre = (
            (ct_array_pre - np.min(ct_array_pre) + epsilon)
            / (np.max(ct_array_pre) - np.min(ct_array_pre) + epsilon)
            * 255.0
        )
        ct_array_pre[ct_array == 0] = 0
    return np.uint8(ct_array_pre)


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
