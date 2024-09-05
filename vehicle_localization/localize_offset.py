import pandas as pd
import joblib
import pandas as pd
import numpy as np
from DSM import get_points_to_cam_angles, project_points_to_ground
import os
from vehicle_localization.utils import *


class OffsetLocalizer:
    @staticmethod
    def _load_data(args) -> pd.DataFrame:
        file_path = args.DetectionPkl
        file_name = os.path.basename(file_path)
        cam_key = get_cam_key(file_name)

        detections = df_from_pickle(file_path)
        detections["width"] = detections["x2"] - detections["x1"]
        detections["height"] = detections["y2"] - detections["y1"]

        detections["center_x"] = detections["x1"] + detections["width"] / 2
        detections["center_y"] = detections["y1"] + detections["height"] / 2

        cent_points = [
            [row["center_x"], row["center_y"]] for _, row in detections.iterrows()
        ]
        angles = get_points_to_cam_angles(args, cent_points, cam_key)
        points_angles_df = pd.DataFrame(
            angles,
            columns=["elevation", "azimuth"],
        )

        output_df = pd.concat(
            [detections, points_angles_df],
            axis=1,
        )
        return output_df

    @staticmethod
    def _run_inference(model_path: str, detections: pd.DataFrame) -> np.ndarray:
        X_data = detections[["width", "height", "elevation", "azimuth"]]
        model = joblib.load(model_path)
        y_pred = model.predict(X_data)

        return y_pred

    @staticmethod
    def localize(args):
        file_path = args.DetectionPkl
        file_name = os.path.basename(file_path)
        cam_key = get_cam_key(file_name)

        detections_data = OffsetLocalizer._load_data(args)
        pred_offsets = OffsetLocalizer._run_inference(
            args.OffsetModelPath, detections_data
        )

        res_x = detections_data["center_x"] - pred_offsets[:, 0]
        res_y = detections_data["center_y"] - pred_offsets[:, 1]

        points_cam = [[point_x, point_y] for point_x, point_y in zip(res_x, res_y)]

        points_world = project_points_to_ground(args, points_cam, cam_key)[0]

        points_world_df = pd.DataFrame(
            points_world,
            columns=["latitude", "longitude"],
        )
        output_df = pd.concat(
            [detections_data[["fn"]], points_world_df],
            axis=1,
        )
        output_df.columns = ["fn", "latitude", "longitude"]
        os.makedirs(args.LocalizationExportPath)
        output_df.to_pickle(
            os.path.join(
                args.LocalizationExportPath, f"{file_name[:-4]}.offset_localized.pkl"
            )
        )

        return "Finished offset localization"
