import numpy as np
import utm

# from IPython.display import display
import pickle
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import cv2
import pickle
import haversine as hs
from haversine import Unit
from itertools import combinations
import rasterio
import json
import os
import math

DATA_DIR = "./data"
DSM_DIR = os.path.join(DATA_DIR, "dsm")
CALIBRATION_DIR = os.path.join(DATA_DIR, "calibrations")
FRAMES_DIR = os.path.join(DATA_DIR, "frames")
ORTHOPHOTO_DIR = os.path.join(DATA_DIR, "orthophoto")
VIS_DIR = os.path.join(DATA_DIR, "visualizations")
KEYPOINTS_VIS_DIR = os.path.join(VIS_DIR, "key_points")
KEYPOINTS_PATH = os.path.join(DATA_DIR, "gt_key_points.csv")
INTRINSICS_PATH = os.path.join(CALIBRATION_DIR, "intrinsic_calibrations.json")
EXTRINSICS_PATH = os.path.join(CALIBRATION_DIR, "extrinsic_calibrations_seg_17.json")
DSM_POINTS_PATH = os.path.join(DSM_DIR, "dsm_points_aoi.xyz")
ORTHOPHOTO_PATH = os.path.join(ORTHOPHOTO_DIR, "orthophoto_window.tif")


def load_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        projection_dict = json.load(f)
    return projection_dict


def load_orthophoto(orthophoto_path):
    """
    Return the georeferenced object and its corresponding numpy object.
    """
    orthophoto_win_tif_obj = rasterio.open(orthophoto_path)
    orthophoto_win_np_obj = np.transpose(
        orthophoto_win_tif_obj.read()[:3, ...], [1, 2, 0]
    ).copy()

    return orthophoto_win_tif_obj, orthophoto_win_np_obj


def load_dsm_points(args):
    """
    This dsm file includes the 3D coordinats from the DSM files that Juan has generated.
    This is a numpy array that includes the 3D coordinates
    """
    dsm_xyz = open(args.DSM_POINTS_PATH)
    coords = []
    for line in dsm_xyz:
        coords.append([float(i) for i in line.split()])

    coords = np.array(coords)
    print("dsm points loaded")

    return coords


def load_cam_frame_dict():
    return {
        "sc1": cv2.imread(os.path.join(FRAMES_DIR, "{}_seg_17_0001.png".format("sc1"))),
        "sc2": cv2.imread(os.path.join(FRAMES_DIR, "{}_seg_17_0001.png".format("sc2"))),
        "sc3": cv2.imread(os.path.join(FRAMES_DIR, "{}_seg_17_0001.png".format("sc3"))),
        "sc4": cv2.imread(os.path.join(FRAMES_DIR, "{}_seg_17_0001.png".format("sc4"))),
    }


def visualize_gt_keypoints(cam_img_dict, orthophoto_win_np_obj, gt_point_df):
    for k in cam_img_dict.keys():
        cam_gt_point_df = gt_point_df[~gt_point_df[k].isnull()]
        for row_idx, row in cam_gt_point_df.iterrows():
            row_eval = eval(row[k])
            cam_img_dict[k] = cv2.drawMarker(
                cv2.putText(
                    cam_img_dict[k],
                    str(row_idx),
                    (int(row_eval[0]), int(row_eval[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                ),
                (int(row_eval[0]), int(row_eval[1])),
                color=(255, 0, 0),
                thickness=2,
                markerType=cv2.MARKER_TILTED_CROSS,
                line_type=cv2.LINE_AA,
                markerSize=15,
            )

        cv2.imwrite(
            os.path.join(KEYPOINTS_VIS_DIR, "{}_gt_point.png".format(k)),
            cam_img_dict[k],
        )

    for row_idx, row in gt_point_df.iterrows():
        row_eval = eval(row["ortho"])
        cv2.drawMarker(
            cv2.putText(
                orthophoto_win_np_obj,
                str(row_idx),
                (int(row_eval[0]), int(row_eval[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            ),
            (int(row_eval[0]), int(row_eval[1])),
            color=(0, 0, 255),
            thickness=2,
            markerType=cv2.MARKER_TILTED_CROSS,
            line_type=cv2.LINE_AA,
            markerSize=15,
        )
    cv2.imwrite(
        os.path.join(KEYPOINTS_VIS_DIR, "orthophoto_gt_points.png"),
        cv2.cvtColor(orthophoto_win_np_obj, cv2.COLOR_BGR2RGB),
    )


def create_raster(args, cam_key, coords, projection_dict, intrinsic_dict):
    """
    Iterate over each camera.
    For each 3D coordiante tiplet, find the corresponding image coordinate in the given camera and
    store a tuple that includes the image coordinates and the corresponding index of the 3D coordinates.
    Append the projected coordinates and the corresponding 3D coordinate index.
    ScaleAI had difficulties processing utm coordinates, therefore a translation transformation was
    first applied on the utm coordinates in order to get the 'local_gta' coordinates.
    In this part we transform the world coordinate onto the pixel coordinates based on which cam we chose.
    Check if the projected coordinates are contained between the image boundaries.
    Also check if the points are in the field of view and not behind the camera.
    Store the set of triplets, which include the image coordinates and the corresponding index of the 3D coordinates,
    in a csv file for each camera.

    """
    print("Creating & saving the raster for camera ...")
    img_ground_map = []

    for coord_idx, coord in enumerate(coords):
        local_coord = np.append(
            coord + np.array(projection_dict["T_localgta_gta"])[:-1, -1], [1]
        ).reshape(4, 1)

        cam_coord = np.matmul(
            np.array(projection_dict["T_{}_localgta".format(cam_key)]), local_coord
        )
        projected_coord = np.matmul(
            np.array(intrinsic_dict[cam_key]["intrinsic_matrix"]), cam_coord
        )
        projected_coord = projected_coord[:-1] / projected_coord[-1]

        if (
            all(
                [
                    0
                    <= np.floor(projected_coord[i, 0])
                    <= intrinsic_dict[cam_key]["image_size"][i] - 1
                    for i in range(2)
                ]
            )
            and cam_coord[2] > 0
        ):
            img_ground_map.append((coord, cam_coord, projected_coord))

    pd.DataFrame(
        {
            "x_w": [i[0][0].squeeze() for i in img_ground_map],
            "y_w": [i[0][1].squeeze() for i in img_ground_map],
            "z_w": [i[0][2].squeeze() for i in img_ground_map],
            "x_c": [i[1][0].squeeze() for i in img_ground_map],
            "y_c": [i[1][1].squeeze() for i in img_ground_map],
            "z_c": [i[1][2].squeeze() for i in img_ground_map],
            "u": [i[2][0].squeeze() for i in img_ground_map],
            "v": [i[2][1].squeeze() for i in img_ground_map],
        }
    ).to_csv(args.ToGroundCorrespondance)
    img_ground_df = pd.read_csv(args.ToGroundCorrespondance)

    img_size = intrinsic_dict[cam_key]["image_size"]

    mesh_coords = np.meshgrid(np.arange(img_size[0]), np.arange(img_size[1]))
    mesh_coords = np.stack([mesh_coords[0], mesh_coords[1]], axis=-1).reshape(-1, 2)

    knn = NearestNeighbors(n_neighbors=1).fit(img_ground_df[["u", "v"]].values)
    knn_idx = knn.kneighbors(mesh_coords, return_distance=False).squeeze()
    img_ground_raster = img_ground_df.iloc[knn_idx][
        ["x_w", "y_w", "z_w"]
    ].values.reshape(img_size[1], img_size[0], 3)

    with open(args.ToGroundRaster, "wb") as f:
        pickle.dump(img_ground_raster, f)


def project_to_cam(args, latitude, longitude, cam_key, k=1):
    """
    We transform the coordinates to UTM format as the 3D coordinates are defined in UTM.
    """
    utm_coordinates = utm.from_latlon(latitude, longitude)

    if not os.path.exists(args.ToGroundRaster):
        coords = load_dsm_points(args)
        intrinsic_dict = load_json_file(args.INTRINSICS_PATH)
        projection_dict = load_json_file(args.EXTRINSICS_PATH)
        create_raster(args, cam_key, coords, projection_dict, intrinsic_dict)

    with open(args.ToGroundRaster, "rb") as f:
        img_ground_raster = pickle.load(f)

    knn = NearestNeighbors(n_neighbors=k).fit(
        img_ground_raster[..., :-1].reshape(-1, 2)
    )
    knn_idx = knn.kneighbors([utm_coordinates[:2]], return_distance=False)[0]

    knn_idx = np.unravel_index(knn_idx.squeeze(), img_ground_raster.shape[:2])
    matched_coord = np.array([knn_idx[1], knn_idx[0]])

    return matched_coord


def project_to_ground(args, u, v, cam_key, k=1):
    # u, v == x, y
    u_int, v_int = int(u), int(v)
    if not os.path.exists(args.ToGroundRaster):
        coords = load_dsm_points(args)
        intrinsic_dict = load_json_file(args.INTRINSICS_PATH)
        projection_dict = load_json_file(args.EXTRINSICS_PATH)
        create_raster(args, cam_key, coords, projection_dict, intrinsic_dict)

    with open(args.ToGroundRaster, "rb") as f:
        img_ground_raster = pickle.load(f)

    matched_coord = img_ground_raster[v_int, u_int]
    latitude, longitude = utm.to_latlon(*matched_coord[:2], 17, northern=True)

    return np.array([latitude, longitude]), matched_coord


def project_points_to_ground(args, points, cam_key, k=1):
    if not os.path.exists(args.ToGroundRaster):
        coords = load_dsm_points(args)
        intrinsic_dict = load_json_file(args.INTRINSICS_PATH)
        projection_dict = load_json_file(args.EXTRINSICS_PATH)
        create_raster(args, cam_key, coords, projection_dict, intrinsic_dict)

    with open(args.ToGroundRaster, "rb") as f:
        img_ground_raster = pickle.load(f)

    matched_coords = []
    lat_longs = []
    for u, v in points:
        u_int, v_int = int(u), int(v)
        matched_coord = img_ground_raster[v_int, u_int]
        latitude, longitude = utm.to_latlon(*matched_coord[:2], 17, northern=True)

        matched_coords.append(matched_coord)
        lat_longs.append([latitude, longitude])

    return np.array(lat_longs), np.array(matched_coords)


def get_points_to_cam_angles(args, points, cam_key):
    projection_dict = load_json_file(args.EXTRINSICS_PATH)
    extrinsics = np.array(projection_dict["T_{}_localgta".format(cam_key)])
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]

    # Compute the camera position in world coordinates
    camera_position = -np.dot(R.T, t)

    matched_coord = project_points_to_ground(args, points, cam_key)[1]
    points_coord = matched_coord + np.array(projection_dict["T_localgta_gta"])[:-1, -1]

    angles = []
    for point_coord in points_coord:
        V = [
            (point_coord[0] - camera_position[0]),
            (point_coord[1] - camera_position[1]),
            (point_coord[2] - camera_position[2]),
        ]

        azimuth = np.arctan2(V[1], V[0])

        # Elevation angle (θ)
        elevation = np.arctan2(V[2], np.sqrt(V[0] ** 2 + V[1] ** 2))

        # Convert to degrees if needed
        azimuth_deg = np.degrees(azimuth)
        elevation_deg = np.degrees(elevation)

        angles.append([elevation_deg, azimuth_deg])

    return np.array(angles)


def get_points_to_cam_distance(args, points, cam_key):
    projection_dict = load_json_file(args.EXTRINSICS_PATH)
    extrinsics = np.array(projection_dict["T_{}_localgta".format(cam_key)])
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]

    # Compute the camera position in world coordinates
    camera_position = -np.dot(R.T, t)

    matched_coord = project_points_to_ground(args, points, cam_key)[1]
    points_coord = matched_coord + np.array(projection_dict["T_localgta_gta"])[:-1, -1]

    distances = []
    for point_coord in points_coord:
        V = [
            (point_coord[0] - camera_position[0]),
            (point_coord[1] - camera_position[1]),
            (point_coord[2] - camera_position[2]),
        ]

        distance = math.sqrt(V[0] ** 2 + V[1] ** 2 + V[2] ** 2)
        distances.append(distance)

    return np.array(distances)


def measure_projection_error():
    error_list = []
    orthophoto_win_tif_obj, orthophoto_win_np_obj = load_orthophoto()
    gt_point_df = pd.read_csv(KEYPOINTS_PATH)
    cam_img_dict = load_cam_frame_dict()

    visualize_gt_keypoints(cam_img_dict, orthophoto_win_np_obj, gt_point_df)

    for c1, c2 in list(combinations(cam_img_dict.keys(), 2)):
        cam_gt_point_df = gt_point_df[
            (~gt_point_df[c1].isnull()) & (~gt_point_df[c2].isnull())
        ][[c1, c2, "ortho", "vicinity"]]

        for row_idx, row in cam_gt_point_df.iterrows():
            row_ortho = eval(row["ortho"])
            row_c1, row_c2 = eval(row[c1]), eval(row[c2])

            gt_ground_coord = utm.to_latlon(
                *rasterio.transform.xy(
                    orthophoto_win_tif_obj.transform, row_ortho[1], row_ortho[0]
                ),
                17,
                northern=True
            )

            gt_pix_coord = row_c2
            proj_ground_coord = project_to_ground(*row_c1, c1, verbose=False)[0]
            proj_pix_coord = project_to_cam(*proj_ground_coord, c2, verbose=False)
            print(
                "point idx: {}, {}(gt: {}) -> world (gt: {}, proj: {}) -> {}(gt: {}, proj: {}), pixel error: {}, geolocation error: {}".format(
                    row_idx,
                    c1,
                    row_c1,
                    gt_ground_coord,
                    proj_ground_coord,
                    c2,
                    gt_pix_coord,
                    proj_pix_coord,
                    round(
                        np.sqrt(
                            np.square(proj_pix_coord - np.array(gt_pix_coord)).sum()
                        ),
                        2,
                    ),
                    round(
                        hs.haversine(
                            gt_ground_coord, proj_ground_coord, unit=Unit.METERS
                        ),
                        2,
                    ),
                )
            )
            cv2.drawMarker(
                cam_img_dict[c2],
                proj_pix_coord,
                color=[0, 0, 180],
                thickness=2,
                markerType=cv2.MARKER_TILTED_CROSS,
                line_type=cv2.LINE_AA,
                markerSize=15,
            )
            error_list.append(
                (
                    c1,
                    c2,
                    row_idx,
                    round(
                        np.sqrt(
                            np.square(proj_pix_coord - np.array(gt_pix_coord)).sum()
                        ),
                        2,
                    ),
                    round(
                        hs.haversine(
                            gt_ground_coord, proj_ground_coord, unit=Unit.METERS
                        ),
                        2,
                    ),
                    row["vicinity"],
                )
            )
            orthophoto_proj_idx = orthophoto_win_tif_obj.index(
                *utm.from_latlon(*proj_ground_coord)[:2]
            )
            cv2.drawMarker(
                orthophoto_win_np_obj,
                (orthophoto_proj_idx[1], orthophoto_proj_idx[0]),
                color=[180, 0, 0],
                thickness=2,
                markerType=cv2.MARKER_TILTED_CROSS,
                line_type=cv2.LINE_AA,
                markerSize=15,
            )

            gt_pix_coord = row_c1
            proj_ground_coord = project_to_ground(*row_c2, c2, verbose=False)[0]
            proj_pix_coord = project_to_cam(*proj_ground_coord, c1, verbose=False)
            print(
                "point idx: {}, {}(gt: {}) -> world (gt: {}, proj: {}) -> {}(gt: {}, proj: {}), pixel error: {}, geolocation error: {}".format(
                    row_idx,
                    c2,
                    row_c2,
                    gt_ground_coord,
                    proj_ground_coord,
                    c1,
                    gt_pix_coord,
                    proj_pix_coord,
                    round(
                        np.sqrt(
                            np.square(proj_pix_coord - np.array(gt_pix_coord)).sum()
                        ),
                        2,
                    ),
                    round(
                        hs.haversine(
                            gt_ground_coord, proj_ground_coord, unit=Unit.METERS
                        ),
                        2,
                    ),
                )
            )
            cv2.drawMarker(
                cam_img_dict[c1],
                proj_pix_coord,
                color=[0, 0, 180],
                thickness=2,
                markerType=cv2.MARKER_TILTED_CROSS,
                line_type=cv2.LINE_AA,
                markerSize=15,
            )
            error_list.append(
                (
                    c2,
                    c1,
                    row_idx,
                    round(
                        np.sqrt(
                            np.square(proj_pix_coord - np.array(gt_pix_coord)).sum()
                        ),
                        2,
                    ),
                    round(
                        hs.haversine(
                            gt_ground_coord, proj_ground_coord, unit=Unit.METERS
                        ),
                        2,
                    ),
                    row["vicinity"],
                )
            )
            orthophoto_proj_idx = orthophoto_win_tif_obj.index(
                *utm.from_latlon(*proj_ground_coord)[:2]
            )
            cv2.drawMarker(
                orthophoto_win_np_obj,
                (int(row_ortho[0]), int(row_ortho[1])),
                color=[180, 0, 0],
                thickness=2,
                markerType=cv2.MARKER_TILTED_CROSS,
                line_type=cv2.LINE_AA,
                markerSize=15,
            )

    for k in cam_img_dict.keys():
        cv2.imwrite(
            os.path.join(KEYPOINTS_VIS_DIR, "{}_gt_proj_points.png".format(k)),
            cam_img_dict[k],
        )

    cv2.imwrite(
        os.path.join(KEYPOINTS_VIS_DIR, "orthophoto_gt_proj_points.png"),
        cv2.cvtColor(orthophoto_win_np_obj, cv2.COLOR_BGR2RGB),
    )

    error_df = pd.DataFrame(
        error_list,
        columns=[
            "src_cam",
            "dst_cam",
            "loc_idx",
            "error[pixel]",
            "error[meter]",
            "vicinity",
        ],
    )
    error_df.to_csv(os.path.join(DATA_DIR, "error_report.csv"))

    rows = []
    for g_name, g in error_df.groupby(["src_cam"]):
        rows.append(
            (
                g_name[0],
                "%.2f" % (g["error[pixel]"].mean()),
                "%.2f" % (g["error[pixel]"].std()),
                "%.2f" % (g["error[meter]"].mean()),
                "%.2f" % (g["error[meter]"].std()),
                len(g),
            )
        )
    df = pd.DataFrame(
        rows,
        columns=[
            "src_cam",
            "mean error[pixel]",
            "std[pixel]",
            "mean error[meter]",
            "std[meter]",
            "point pair count",
        ],
    )
    display(df)

    rows = []
    for g_name, g in error_df.groupby(["dst_cam"]):
        rows.append(
            (
                g_name[0],
                "%.2f" % (g["error[pixel]"].mean()),
                "%.2f" % (g["error[pixel]"].std()),
                "%.2f" % (g["error[meter]"].mean()),
                "%.2f" % (g["error[meter]"].std()),
                len(g),
            )
        )
    df = pd.DataFrame(
        rows,
        columns=[
            "dst_cam",
            "mean error[pixel]",
            "std[pixel]",
            "mean error[meter]",
            "std[meter]",
            "point pair count",
        ],
    )
    display(df)

    rows = []
    for g_name, g in error_df.groupby(["vicinity"]):
        if g_name[0] == 1:
            vicinity_str = "insdie"
        elif g_name[0] == 2:
            vicinity_str = "within 100m"
        else:
            vicinity_str = "beyond 100m"
        rows.append(
            (
                vicinity_str,
                "%.2f" % (g["error[pixel]"].mean()),
                "%.2f" % (g["error[pixel]"].std()),
                "%.2f" % (g["error[meter]"].mean()),
                "%.2f" % (g["error[meter]"].std()),
                len(g),
            )
        )
    df = pd.DataFrame(
        rows,
        columns=[
            "vicinity",
            "mean error[pixel]",
            "std[pixel]",
            "mean error[meter]",
            "std[meter]",
            "point pair count",
        ],
    )
    display(df)

    rows = []
    for g_name, g in error_df.groupby(["loc_idx"]):
        if g["vicinity"].iloc[0] == 1:
            vicinity_str = "insdie"
        elif g["vicinity"].iloc[0] == 2:
            vicinity_str = "within 100m"
        else:
            vicinity_str = "beyond 100m"
        rows.append(
            (
                g_name[0],
                "%.2f" % (g["error[pixel]"].mean()),
                "%.2f" % (g["error[pixel]"].std()),
                "%.2f" % (g["error[meter]"].mean()),
                "%.2f" % (g["error[meter]"].std()),
                vicinity_str,
            )
        )
    df = pd.DataFrame(
        rows,
        columns=[
            "loc_idx",
            "error[pixel]",
            "std[pixel]",
            "error[meter]",
            "std[meter]",
            "vicinity",
        ],
    )
    display(df)
