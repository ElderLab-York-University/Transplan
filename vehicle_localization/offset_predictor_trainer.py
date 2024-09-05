from typing import List
import os
import pandas as pd
from pandas import DataFrame
import pickle
import numpy as np
import joblib
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from datetime import datetime
from sklearn.metrics import mean_squared_error, make_scorer
from DSM import get_points_to_cam_angles
from vehicle_localization.utils import *


class OffsetPredictorTrainer:
    @staticmethod
    def _load_detections_data(args) -> DataFrame:
        # lists should contain path of 2d, 3d detection files, ith element in all lists should correspond
        # to the same segment of video

        dets_2d_files = args.Dets2dFilesOffsetTraining
        dets_3d_files = args.Dets3dFilesOffsetTraining

        if len(dets_2d_files) != len(dets_3d_files):
            raise ValueError("input lists should have the same size")

        all_data = []
        all_angles = []
        for path_2d, path_3d in zip(dets_2d_files, dets_3d_files):
            file_name = os.path.basename(path_2d)
            cam_key = get_cam_key(file_name)

            df_2d = df_from_pickle(path_2d)
            df_3d = df_from_pickle(path_3d)

            fn_2d_diff = 0
            fn_3d_diff = 0
            if min(df_2d["fn"].unique()) < 0:
                fn_2d_diff = 0 - min(df_2d["fn"].unique())

            if min(df_3d["fn"].unique()) < 0:
                fn_3d_diff = 0 - min(df_3d["fn"].unique())

            points = []
            for index, row in df_2d.iterrows():
                try:
                    uuid = row["uuid"]
                    fn = row["fn"]

                    x_cent_bbox = int((row["x2"] + row["x1"]) / 2)
                    y_cent_bbox = int((row["y2"] + row["y1"]) / 2)

                    info_3d = get_val_uuid_fn(df_3d, uuid, fn + fn_2d_diff - fn_3d_diff)

                    x_center_act = int(
                        (
                            info_3d["x1"].iloc[0]
                            + info_3d["x2"].iloc[0]
                            + info_3d["x3"].iloc[0]
                            + info_3d["x4"].iloc[0]
                        )
                        / 4
                    )

                    y_center_act = int(
                        (
                            info_3d["y1"].iloc[0]
                            + info_3d["y2"].iloc[0]
                            + info_3d["y3"].iloc[0]
                            + info_3d["y4"].iloc[0]
                        )
                        / 4
                    )

                    bounded_center = bound_point_to_bbox(
                        x_center_act,
                        y_center_act,
                        row["x1"],
                        row["y1"],
                        row["x2"],
                        row["y2"],
                    )

                    points.append([x_cent_bbox, y_cent_bbox])

                    all_data.append(
                        [
                            fn + fn_2d_diff,
                            row["class"],
                            row["x1"],
                            row["y1"],
                            row["x2"],
                            row["y2"],
                            bounded_center[0],
                            bounded_center[1],
                        ]
                    )
                except:
                    continue

            proj_angles = get_points_to_cam_angles(
                args=args,
                points=points,
                cam_key=cam_key,
            )
            all_angles += proj_angles.tolist()

        data_angles = pd.DataFrame(all_angles, columns=["elevation", "azimuth"])
        data_other = pd.DataFrame(
            all_data,
            columns=[
                "fn",
                "class",
                "x1",
                "y1",
                "x2",
                "y2",
                "gt_x_cent_corrected",
                "gt_y_cent_corrected",
            ],
        )

        data = pd.concat(
            [data_angles, data_other],
            axis=1,
        )

        data["width"] = data["x2"] - data["x1"]
        data["height"] = data["y2"] - data["y1"]
        data["x_cent_bbox"] = (data["x2"] + data["x1"]) / 2
        data["y_cent_bbox"] = (data["y2"] + data["y1"]) / 2
        data["x_diff"] = data["x_cent_bbox"] - data["gt_x_cent_corrected"]
        data["y_diff"] = data["y_cent_bbox"] - data["gt_y_cent_corrected"]

        return data

    @staticmethod
    def train_knn_reg(args):
        train_data = OffsetPredictorTrainer._load_detections_data(args)
        X_train = train_data[["width", "height", "elevation", "azimuth"]]
        y_train = train_data[["x_diff", "y_diff"]]

        knn = KNeighborsRegressor()
        param_grid = {
            "estimator__n_neighbors": [3, 5, 7, 9],  # Example parameter grid for KNN
            "estimator__weights": ["uniform", "distance"],
            "estimator__metric": ["euclidean", "manhattan"],
        }

        multi_output_knn = MultiOutputRegressor(knn)
        grid_search = GridSearchCV(
            multi_output_knn,
            param_grid,
            scoring=make_scorer(mean_squared_error, greater_is_better=False),
            cv=5,
        )
        grid_search.fit(X_train, y_train)
        knn_best_model = grid_search.best_estimator_
        joblib.dump(
            knn_best_model,
            args.OffsetModelPath,
        )

    @staticmethod
    def train_linear_reg(args):
        train_data = OffsetPredictorTrainer._load_detections_data(args)
        X_train = train_data[["width", "height", "elevation", "azimuth"]]
        y_train = train_data[["x_diff", "y_diff"]]

        lr_model = MultiOutputRegressor(LinearRegression())
        lr_model.fit(X_train, y_train)
        joblib.dump(
            lr_model,
            args.OffsetModelPath,
        )

    @staticmethod
    def train_random_forest_reg(args):
        train_data = OffsetPredictorTrainer._load_detections_data(args)
        X_train = train_data[["width", "height", "elevation", "azimuth"]]
        y_train = train_data[["x_diff", "y_diff"]]

        rf = RandomForestRegressor(max_depth=10, min_samples_split=2, n_estimators=100)
        multi_output_rf = MultiOutputRegressor(rf)
        multi_output_rf.fit(X_train, y_train)
        joblib.dump(
            multi_output_rf,
            args.OffsetModelPath,
        )

        return f"Trained and saved {args.OffsetModelPath}"
