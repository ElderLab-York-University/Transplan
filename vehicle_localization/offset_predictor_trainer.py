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
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
import argparse


def get_all_file_paths(directories: List[str]) -> List[str]:
    all_files = []
    for dir_path in directories:
        for root, _, files in os.walk(dir_path):
            for file in files:
                full_path = os.path.join(root, file)
                all_files.append(full_path)
    return all_files


def read_pkl(file_path: str) -> DataFrame:
    try:
        with open(file_path, "rb") as file:
            data = pickle.load(file)
            df = pd.DataFrame(data)

            return df
    except Exception as e:
        raise ValueError(f"An error occurred: {e}")


def get_specific_val(data, uuid, fn) -> DataFrame:
    try:
        df = pd.DataFrame(data)
        filtered_df = df[df["uuid"] == uuid]
        return filtered_df[filtered_df["fn"] == fn]
    except Exception as e:
        raise ValueError(f"An error occurred: {e}")


def bound_point_to_bbox(x, y, x1, y1, x2, y2):
    # Ensure x1 <= x2 and y1 <= y2 for a valid bbox
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)

    # Bound the x and y coordinates
    bounded_x = max(x_min, min(x, x_max))
    bounded_y = max(y_min, min(y, y_max))

    return bounded_x, bounded_y


def prject_point_mtarix_DSM(img_ground_raster, u, v):
    x, y, z = img_ground_raster[v, u]
    return [x - 629540, y - 4855460, z]


def get_proj_angles(cam_name, u, v, img_ground_raster):
    if "sc1" in cam_name:
        sc = "sc1"
        K = np.array(
            [[2160.0, 0.0, 2383.744763], [0.0, 2160.0, 1328.069528], [0.0, 0.0, 1.0]]
        )
        extrinsics_4 = np.array(
            [
                [
                    -0.30752332659441584,
                    -0.9513509870424467,
                    0.018992182547011478,
                    535.7222453982591,
                ],
                [
                    -0.38069365811525124,
                    0.10471746163700583,
                    -0.9187527370783074,
                    330.1414676005732,
                ],
                [
                    0.8720675101183483,
                    -0.2897681014443422,
                    -0.3943763495243638,
                    -203.21173521244773,
                ],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    elif "sc2" in cam_name:
        sc = "sc2"
        K = np.array(
            [
                [2160.0, 0.0, 2443.555595],
                [0.0, 2160.0, 1349.428111],
                [0.0, 0.0, 1.0],
            ]
        )
        extrinsics_4 = np.array(
            [
                [
                    0.8960545651056627,
                    -0.4414303579389249,
                    -0.047174733345142564,
                    -254.27569360873514,
                ],
                [
                    -0.2149757919887656,
                    -0.3384764591377275,
                    -0.9160890215811138,
                    424.67727110770846,
                ],
                [
                    0.38842196797366396,
                    0.831007175496346,
                    -0.3981902171548002,
                    -412.23779378808354,
                ],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    elif "sc3" in cam_name:
        sc = "sc3"
        K = np.array(
            [
                [2160.0, 0.0, 2435.560499],
                [0.0, 2160.0, 1336.25684],
                [0.0, 0.0, 1.0],
            ]
        )
        extrinsics_4 = np.array(
            [
                [
                    -0.9024559816748954,
                    0.43018588669720986,
                    -0.022656216266661377,
                    281.91169335836634,
                ],
                [
                    0.18091772254361593,
                    0.3307542263555821,
                    -0.9262129449996842,
                    -38.81437941190972,
                ],
                [
                    -0.3909500976349825,
                    -0.8399653234751452,
                    -0.3763193809830071,
                    643.0919055847118,
                ],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    elif "sc4" in cam_name:
        sc = "sc4"
        K = np.array(
            [
                [2160.0, 0.0, 2402.968841],
                [0.0, 2160.0, 1357.883747],
                [0.0, 0.0, 1.0],
            ]
        )
        extrinsics_4 = np.array(
            [
                [
                    0.3324242805775379,
                    0.9428731401467102,
                    -0.02200770805825944,
                    -536.8463108464244,
                ],
                [
                    0.3779979036480696,
                    -0.1545749433505824,
                    -0.9128111370129997,
                    56.00011384515877,
                ],
                [
                    -0.8640669433313272,
                    0.2951217180011799,
                    -0.4077885347017972,
                    425.49930478163765,
                ],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    else:
        raise ValueError("invalid camera name")

    p = prject_point_mtarix_DSM(img_ground_raster, u, v)
    R = extrinsics_4[:3, :3]
    t = extrinsics_4[:3, 3]

    # Compute the camera position in world coordinates
    camera_position_world = -np.dot(R.T, t)

    V = [
        (p[0] - camera_position_world[0]),
        (p[1] - camera_position_world[1]),
        (p[2] - camera_position_world[2]),
    ]

    azimuth = np.arctan2(V[1], V[0])

    # Elevation angle (θ)
    elevation = np.arctan2(V[2], np.sqrt(V[0] ** 2 + V[1] ** 2))

    # Convert to degrees if needed
    azimuth_deg = np.degrees(azimuth)
    elevation_deg = np.degrees(elevation)

    return {"elevation": elevation_deg, "azimuth": azimuth_deg}


class OffsetPredictorTrainer:
    @staticmethod
    def load_detections_data(
        dets_2d_files: List[str], dets_3d_files: List[str], DSM_files: List[str]
    ) -> DataFrame:
        # lists should contain path of 2d, 3d detection files as well as DSM files, ith element in all lists should correspond
        # to the same segment of video
        if len(dets_2d_files) != len(dets_3d_files) or len(dets_2d_files) != len(
            DSM_files
        ):
            raise ValueError("input lists should have the same size")

        all_data = []
        for path_2d, path_3d, path_DSM in zip(dets_2d_files, dets_3d_files, DSM_files):
            df_2d = read_pkl(path_2d)
            df_3d = read_pkl(path_3d)
            with open(path_DSM, "rb") as f:
                img_ground_raster = pickle.load(f)
            fn_2d_diff = 0
            fn_3d_diff = 0
            if min(df_2d["fn"].unique()) < 0:
                fn_2d_diff = 0 - min(df_2d["fn"].unique())

            if min(df_3d["fn"].unique()) < 0:
                fn_3d_diff = 0 - min(df_3d["fn"].unique())

            for index, row in df_2d.iterrows():
                try:
                    uuid = row["uuid"]
                    fn = row["fn"]

                    x_cent_bbox = int((row["x2"] + row["x1"]) / 2)
                    y_cent_bbox = int((row["y2"] + row["y1"]) / 2)

                    info_3d = get_specific_val(
                        df_3d, uuid, fn + fn_2d_diff - fn_3d_diff
                    )

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

                    proj_angles = get_proj_angles(
                        os.path.basename(path_2d),
                        x_cent_bbox,
                        y_cent_bbox,
                        img_ground_raster,
                    )
                    all_data.append(
                        [
                            proj_angles["elevation"],
                            proj_angles["azimuth"],
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
        data = pd.DataFrame(
            all_data,
            columns=[
                "elevation",
                "azimuth",
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

        data["width"] = data["x2"] - data["x1"]
        data["height"] = data["y2"] - data["y1"]
        data["x_cent_bbox"] = (data["x2"] + data["x1"]) / 2
        data["y_cent_bbox"] = (data["y2"] + data["y1"]) / 2
        data["x_diff"] = data["x_cent_bbox"] - data["gt_x_cent_corrected"]
        data["y_diff"] = data["y_cent_bbox"] - data["gt_y_cent_corrected"]

        return data

    @staticmethod
    def train_knn_reg(train_data: DataFrame, save_path: str):
        X_train = train_data["width", "height", "elevation", "azimuth"]
        y_train = train_data["x_diff", "y_diff"]

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
            os.path.join(
                save_path,
                f"offset_knn_model_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.pkl",
            ),
        )

    @staticmethod
    def train_linear_reg(train_data: DataFrame, save_path: str):
        X_train = train_data["width", "height", "elevation", "azimuth"]
        y_train = train_data["x_diff", "y_diff"]

        lr_model = MultiOutputRegressor(LinearRegression())
        lr_model.fit(X_train, y_train)
        joblib.dump(
            lr_model,
            os.path.join(
                save_path,
                f"offset_lr_model_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.pkl",
            ),
        )

    @staticmethod
    def train_random_forest_reg(train_data: DataFrame, save_path: str):
        X_train = train_data["width", "height", "elevation", "azimuth"]
        y_train = train_data["x_diff", "y_diff"]

        rf = RandomForestRegressor(max_depth=10, min_samples_split=2, n_estimators=100)
        multi_output_rf = MultiOutputRegressor(rf)
        multi_output_rf.fit(X_train, y_train)
        joblib.dump(
            multi_output_rf,
            os.path.join(
                save_path,
                f"offset_rf_model_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.pkl",
            ),
        )


def main():
    parser = argparse.ArgumentParser(description="Train offset prediction models.")

    parser.add_argument(
        "--dets_2d_files", nargs="+", help="path of 2d detection pkl files for training"
    )
    parser.add_argument(
        "--dets_3d_files",
        nargs="+",
        help="path of corresponding 3d detection pkl files for training",
    )
    parser.add_argument(
        "--DSM_files", nargs="+", help="path of corresponding DSM files for training"
    )

    parser.add_argument(
        "--train_knn_reg", action="store_true", help="enable to train knn model"
    )
    parser.add_argument(
        "--train_lin_reg", action="store_true", help="enable to train linear model"
    )
    parser.add_argument(
        "--train_rf_reg",
        action="store_true",
        help="enable to train random forest model",
    )

    parser.add_argument(
        "--models_save_path", type=str, help="path to save the models", default="models"
    )

    args = parser.parse_args()

    train_data = OffsetPredictorTrainer.load_detections_data(
        dets_2d_files=args.dets_2d_files,
        dets_3d_files=args.dets_3d_files,
        DSM_files=args.DSM_files,
    )

    os.makedirs(args.models_save_path, exist_ok=True)

    if args.train_lin_reg:
        OffsetPredictorTrainer.train_linear_reg(
            train_data=train_data, save_path=args.models_save_path
        )
    if args.train_knn_reg:
        OffsetPredictorTrainer.train_knn_reg(
            train_data=train_data, save_path=args.models_save_path
        )
    if args.train_rf_reg:
        OffsetPredictorTrainer.train_random_forest_reg(
            train_data=train_data, save_path=args.models_save_path
        )


if __name__ == "__main__":
    main()
