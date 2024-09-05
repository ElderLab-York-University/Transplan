from vehicle_localization.depth_anything_v2_inference import DepthAnythingV2Inference
from DSM import get_points_to_cam_distance
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import cv2
import numpy as np
import joblib


class DepthScalaerTrainer:
    def __init__(self, args) -> None:
        self.model = DepthAnythingV2Inference()
        self.args = args

    def train(self):
        points = self.args.DepthScalingGroundPoints
        tupled_points = [(points[i], points[i + 1]) for i in range(0, len(points), 2)]
        img_path = self.args.DepthScalingImgPath

        img = cv2.imread(img_path)
        relative_depth_map = self.model.run_inference(img)

        relative_depths = np.array(
            [relative_depth_map[int(v), int(u)] for u, v in tupled_points]
        )

        # Known metric depths
        metric_depths = get_points_to_cam_distance(
            self.args, tupled_points, self.args.CamKey
        )

        # Create polynomial features
        poly = PolynomialFeatures(degree=2)
        relative_depths_poly = poly.fit_transform(relative_depths.reshape(-1, 1))

        # Fit polynomial regression model
        model = LinearRegression().fit(relative_depths_poly, metric_depths)

        joblib.dump((model, poly), self.args.DepthScalingModelPath)

        return f"Trained and saved {self.args.DepthScalingModelPath}"
