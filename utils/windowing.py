import numpy as np
import json
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
