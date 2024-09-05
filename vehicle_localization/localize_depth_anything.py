from vehicle_localization.depth_anything_v2_inference import DepthAnythingV2Inference
from DSM import project_points_to_ground, load_json_file
import cv2
import numpy as np
import joblib
import pandas as pd
from vehicle_localization.utils import *
import utm
import os
from vehicle_localization.utils import bound_point_to_bbox


class DepthAnythingLocalizer:
    def __init__(self, args) -> None:
        self.model = DepthAnythingV2Inference()
        self.args = args
        self.intrinsic_dict = load_json_file(args.INTRINSICS_PATH)
        self.projection_dict = load_json_file(args.EXTRINSICS_PATH)

    @staticmethod
    def _load_model_and_apply_to_depth_map(relative_depth_map, model_path):
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
                row["fn"],
            )
            if frame_number not in frames_dict:
                frames_dict[frame_number] = []
            frames_dict[frame_number].append(bbox)

        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        ground_points = []
        bbox_points = []
        frames_imgs = {}
        frames_depth = {}
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count in frames_dict:
                frames_imgs[frame_count] = frame

            frame_count += 1

        cap.release()

        # run deoth anything on all frames
        fns = list(frames_imgs.keys())
        frames = list(frames_imgs.values())
        batch_size = 8
        depth_imgs_relative = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i : i + batch_size]
            depth_imgs_relative += self.model.run_batch_inference(batch)

        frames_depth = {
            fn: np.array(depth_img["depth"])
            for fn, depth_img in zip(fns, depth_imgs_relative)
        }

        # create point cloud of each frame and find point coords
        for fn, depth_img_relative in frames_depth.items():
            # depth_img_relative = self.model.run_inference(frame)
            depth_img = DepthAnythingLocalizer._load_model_and_apply_to_depth_map(
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

            frame_shape = frames_imgs[fn].shape

            bbox_infos = frames_dict[frame_count]
            for bbox in bbox_infos:
                x1, y1, x2, y2, fn = bbox
                cent_u = int((x1 + x2) / 2)
                cent_v = int((y1 + y2) / 2)
                point_3d = DepthAnythingLocalizer._get_3d_coordinates_from_point_cloud(
                    cent_u,
                    cent_v,
                    point_cloud_non_ground,
                    frame_shape,
                    self.projection_dict,
                )

                bbox_points.append([x1, y1])
                bbox_points.append([x2, y2])

                latitude, longitude = utm.to_latlon(*point_3d[:2], 17, northern=True)
                ground_points.append([fn, latitude, longitude])

        # bound points to projected bbox
        projected_bbox_points = project_points_to_ground(
            self.args, bbox_points, self.args.CamKey
        )[0].tolist()

        grouped_dprojected_bbox_points = [
            projected_bbox_points[i : i + 2]
            for i in range(0, len(projected_bbox_points), 2)
        ]
        corrected_ground_points = []
        for index, bbox_points in enumerate(grouped_dprojected_bbox_points):
            point_3d_corrected = bound_point_to_bbox(
                ground_points[index][1],
                ground_points[index][2],
                bbox_points[0][0],
                bbox_points[0][1],
                bbox_points[1][0],
                bbox_points[1][1],
            )
            corrected_ground_points.append(
                [ground_points[index][0], point_3d_corrected[0], point_3d_corrected[1]]
            )

        output_df = pd.DataFrame(
            corrected_ground_points, columns=["fn", "latitude", "longitude"]
        )
        os.makedirs(self.args.LocalizationExportPath)
        output_df.to_pickle(
            os.path.join(
                self.args.LocalizationExportPath,
                f"{file_name[:-4]}.depth_anything_localized.pkl",
            )
        )

        return "Finished depth localization"
