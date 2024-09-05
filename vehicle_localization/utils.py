import pandas as pd
from typing import Tuple


def df_from_pickle(pickle_path: str) -> pd.DataFrame:
    return pd.read_pickle(pickle_path)


def bound_point_to_bbox(x, y, x1, y1, x2, y2) -> Tuple:
    # Ensure x1 <= x2 and y1 <= y2 for a valid bbox
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)

    # Bound the x and y coordinates
    bounded_x = max(x_min, min(x, x_max))
    bounded_y = max(y_min, min(y, y_max))

    return bounded_x, bounded_y


def get_cam_key(file_name: str) -> str:
    cam_keys = ["sc1", "sc2", "sc3", "sc4", "lc1", "lc2"]
    for key in cam_keys:
        if key in file_name:
            return key
    return ""


def get_val_uuid_fn(data, uuid, fn) -> pd.DataFrame:
    try:
        df = pd.DataFrame(data)
        filtered_df = df[df["uuid"] == uuid]
        return filtered_df[filtered_df["fn"] == fn]
    except Exception as e:
        raise ValueError(f"An error occurred: {e}")
