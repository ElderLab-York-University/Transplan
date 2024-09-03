from vehicle_localization.depth_anything_v2_inference import DepthAnythingV2Inference
from DSM import project_to_ground, load_json_file
import cv2
import numpy as np
import joblib
import pandas as pd
from vehicle_localization.utils import *
import utm
import os


class DepthAnythingLocalizer:
    def __init__(self, args) -> None:
        self.model = DepthAnythingV2Inference()
        self.args = args
        self.intrinsic_dict = load_json_file(args.INTRINSICS_PATH)
        self.projection_dict = load_json_file(args.EXTRINSICS_PATH)

    @staticmethod
    def _load_model_and_apply_to_point_cloud(relative_depth_map, model_path):
        # Load the model and the polynomial feature transformer
        model, poly = joblib.load(model_path)

        # Apply the model to the entire relative depth map
        relative_depth_map_flat = relative_depth_map.flatten()
        relative_depth_map_poly = poly.transform(relative_depth_map_flat.reshape(-1, 1))
        metric_depth_map_flat = model.predict(relative_depth_map_poly)
        metric_depth_map = metric_depth_map_flat.reshape(relative_depth_map.shape)

        return metric_depth_map

    @staticmethod
    def _depth_to_point_cloud(depth_map, K, Tcw):
        h, w = depth_map.shape
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # Create a grid of (u, v) coordinates
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        # Calculate X_c, Y_c, Z_c
        X_c = (u - cx) * depth_map / fx
        Y_c = (v - cy) * depth_map / fy
        Z_c = depth_map

        # Stack into (X_c, Y_c, Z_c, 1) format
        points_cam = np.stack((X_c, Y_c, Z_c, np.ones_like(Z_c)), axis=-1)
        points_cam = points_cam.reshape(-1, 4).T  # Shape (4, N)

        # Transform to world coordinates
        points_world = Tcw @ points_cam

        return points_world[:3].T  # Shape (N, 3)

    @staticmethod
    def _get_3d_coordinates_from_point_cloud(
        u, v, point_cloud, image_shape, projection_dict
    ):
        # Calculate the index in the flattened point cloud array
        index = v * image_shape[1] + u

        # Extract the corresponding 3D coordinates
        world_coordinates = point_cloud[index]

        return world_coordinates + np.array(projection_dict["T_gta_localgta"])[:-1, -1]

    def localize(self):
        file_path = self.args.DetectionPkl
        file_name = os.path.basename(file_path)
        video_path = self.args.Video

        detections = df_from_pickle(self.args.DetectionPkl)
        fn_diff = 0
        if min(detections["fn"].unique()) < 0:
            fn_diff = 0 - min(detections["fn"].unique())

        frames_dict = {}

        for i, (_, row) in enumerate(detections.iterrows()):
            frame_number = int(row["fn"]) + fn_diff
            bbox = (
                int(row["x1"]),
                int(row["y1"]),
                int(row["x2"]),
                int(row["y2"]),
                row["uuid"],
                row["fn"],
            )
            if frame_number not in frames_dict:
                frames_dict[frame_number] = []
            frames_dict[frame_number].append(bbox)

        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        ground_points = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count in frames_dict:
                depth_img_relative = self.model.run_inference(frame)
                depth_img = DepthAnythingLocalizer._load_model_and_apply_to_point_cloud(
                    depth_img_relative,
                    model_path=self.args.DepthScalingModelPath,
                )
                point_cloud_non_ground = DepthAnythingLocalizer._depth_to_point_cloud(
                    depth_img,
                    np.array(self.intrinsic_dict[self.args.CamKey]["intrinsic_matrix"]),
                    np.array(
                        self.projection_dict["T_{}_localgta".format(self.args.CamKey)]
                    ),
                )

                frame_shape = frame.shape

                bbox_infos = frames_dict[frame_count]
                for bbox in bbox_infos:
                    x1, y1, x2, y2, uuid, fn = bbox
                    cent_u = int((x1 + x2) / 2)
                    cent_v = int((y1 + y2) / 2)
                    point_3d = (
                        DepthAnythingLocalizer._get_3d_coordinates_from_point_cloud(
                            cent_u,
                            cent_v,
                            point_cloud_non_ground,
                            frame_shape,
                            self.projection_dict,
                        )
                    )

                    xy1_3d = project_to_ground(self.args, x1, y1, self.args.CamKey)[1]

                    xy2_3d = project_to_ground(self.args, x2, y2, self.args.CamKey)[1]

                    point_3d_corrected = bound_point_to_bbox(
                        point_3d[0],
                        point_3d[1],
                        xy1_3d[0],
                        xy1_3d[1],
                        xy2_3d[0],
                        xy2_3d[1],
                    )

                    latitude, longitude = utm.to_latlon(
                        *point_3d_corrected[:2], 17, northern=True
                    )
                    ground_points.append([uuid, fn, latitude, longitude])

            frame_count += 1

        cap.release()

        output_df = pd.DataFrame(
            ground_points, columns=["uuid", "fn", "latitude", "longitude"]
        )
        os.makedirs(self.args.LocalizationExportPath)
        output_df.to_pickle(
            os.path.join(
                self.args.LocalizationExportPath,
                f"{file_name[:-4]}.depth_anything_localized.pkl",
            )
        )

        return "Finished depth localization"
