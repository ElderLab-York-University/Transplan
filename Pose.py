from Libs import *
from Utils import *

# import all posers here
# and add poser names to the "posers" dictionary
# -----------------------------------------------
import Posers.MMPose.extract
# -----------------------------------------------
posers  = {}
posers["MMPose"] = Posers.MMPose.extract
# -----------------------------------------------

def extract_pose(args):
    if args.Poser not in os.listdir("./Posers/"):
        return FailLog("Poser not recognized in ./Posers/")
    current_poser = posers[args.Poser]
    current_poser.extract(args)
    return SucLog("Pose Estimation was successful")

def vis_pose(args):
    pose_keys = [f"k{i}" for i in range(17)]
    pose_df = pd.read_pickle(args.PosePth)
    output_path = args.PoseVisPth
    pose_th = args.PoseTh

    cap = cv2.VideoCapture(args.Video)
    if (cap.isOpened()== False): 
        return FailLog("Error opening video stream or file")
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    out_cap = cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width,frame_height))

    if  not args.ForNFrames is None:
        frames = int(min(args.ForNFrames, frames))

    for frame_num in tqdm(range(frames)):
        if (not cap.isOpened()):
            return FailLog("Error reading the video")
        ret, frame = cap.read()
        if not ret: continue

        pose_result = []
        for i, row in pose_df[pose_df["fn"]==frame_num].iterrows():
            kpts = [i for i in row[pose_keys].to_numpy()]
            pose_result.append(kpts)
            frame = draw_box_on_image(frame, row["x1"], row["y1"], row["x2"], row["y2"])
            if "CP" in pose_df.keys():
                x_cp , y_cp = row["CP"][0], row["CP"][1]
                cv2.circle(frame, (int(x_cp), int(y_cp)), 4,
                                [0, 0, 255], -1)
            if "ContactPoint" in pose_df.keys():
                x_cp , y_cp = row["ContactPoint"][0], row["ContactPoint"][1]
                cv2.circle(frame, (int(x_cp), int(y_cp)), 4,
                                [0, 0, 0], -1)
        pose_result = np.array(pose_result)
        
        frame = draw_keypoints(frame, pose_result, kpt_score_thr=pose_th)
        out_cap.write(frame)
    cap.release()
    out_cap.release()
    return SucLog("Pose visualized on the video")

def vis_detect_top_mc(args, args_mc):
    save_path = args.VisDetectTopMCPth

    cap = cv2.VideoCapture(args_mc[0].Video)
    # Check if camera opened successfully
    if (cap.isOpened()== False): return FailLog("could not open input video")

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()


    second_image_path = args_mc[0].HomographyTopView
    img2 = cv.imread(second_image_path)
    rows2, cols2, dim2 = img2.shape
    frame_width , frame_height  = cols2 , rows2

    dfs =[]
    cam_ids = []
    for arg in args_mc:
        df = pd.read_pickle(arg.PosePth)
        dfs.append(df)
        cam_ids.append(arg.CamID)

    out_cap = cv2.VideoWriter(save_path,cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width,frame_height))
    color = (0, 0, 102)

    if not args.ForNFrames is None:
        frames = int(min(frames, args.ForNFrames))
    # Read until video is completed
    for frame_num in tqdm(range(frames)):
        img2_fn = copy.deepcopy(img2)
        for df, cam_id in zip(dfs, cam_ids):
            this_frame_tracks = df[df.fn==(frame_num)]
            for i, track in this_frame_tracks.iterrows():
                # plot the bbox + id with colors
                x , y  = int(track.x), int(track.y)
                np.random.seed(int(cam_id[1:]))
                color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
                img2_fn = cv.circle(img2_fn, (x,y), radius=3, color=color, thickness=3)

                # cv2.putText(img2_fn, f'cam:', (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
                cv2.putText(img2_fn, f'{cam_id}', (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

        out_cap.write(img2_fn)
    out_cap.release()

    return SucLog("vist detection (using estimated CP) multi camera done.")

def draw_box_on_image(img, x1, y1, x2, y2, c=(0, 0, 255), thickness=1):
    sta_point = (int(x1), int(y1))
    end_point = (int(x2), int(y2))
    img = cv2.rectangle(img, sta_point, end_point, c, thickness)
    return img

def draw_keypoints(img,
                     pose_result,
                     skeleton=None,
                     kpt_score_thr=0.3,
                     pose_kpt_color=None,
                     pose_link_color=None,
                     radius=4,
                     thickness=1,
                     show_keypoint_weight=False):
    """Draw keypoints and links on an image.

    Args:
            img (str or Tensor): The image to draw poses on. If an image array
                is given, id will be modified in-place.
            pose_result (list[kpts]): The poses to draw. Each element kpts is
                a set of K keypoints as an Kx3 numpy.ndarray, where each
                keypoint is represented as x, y, score.
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints. If None,
                the keypoint will not be drawn.
            pose_link_color (np.array[Mx3]): Color of M links. If None, the
                links will not be drawn.
            thickness (int): Thickness of lines.
    """
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                            [230, 230, 0], [255, 153, 255], [153, 204, 255],
                            [255, 102, 255], [255, 51, 255], [102, 178, 255],
                            [51, 153, 255], [255, 153, 153], [255, 102, 102],
                            [255, 51, 51], [153, 255, 153], [102, 255, 102],
                            [51, 255, 51], [0, 255, 0], [0, 0, 255],
                            [255, 0, 0], [255, 255, 255]])

    skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
                [3, 5], [4, 6]]

    pose_link_color = palette[[
        0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
    ]]

    pose_kpt_color = palette[[
        16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0
    ]]

    img_h, img_w, _ = img.shape

    for kpts in pose_result:

        kpts = np.array(kpts, copy=False)

        # draw each point on image
        if pose_kpt_color is not None:
            assert len(pose_kpt_color) == len(kpts)

            for kid, kpt in enumerate(kpts):
                x_coord, y_coord, kpt_score = int(kpt[0]), int(kpt[1]), kpt[2]

                if kpt_score < kpt_score_thr or pose_kpt_color[kid] is None:
                    # skip the point that should not be drawn
                    continue

                color = tuple(int(c) for c in pose_kpt_color[kid])
                if show_keypoint_weight:
                    img_copy = img.copy()
                    cv2.circle(img_copy, (int(x_coord), int(y_coord)), radius,
                               color, -1)
                    transparency = max(0, min(1, kpt_score))
                    cv2.addWeighted(
                        img_copy,
                        transparency,
                        img,
                        1 - transparency,
                        0,
                        dst=img)
                else:
                    cv2.circle(img, (int(x_coord), int(y_coord)), radius,
                               color, -1)

        # draw links
        if skeleton is not None and pose_link_color is not None:
            assert len(pose_link_color) == len(skeleton)

            for sk_id, sk in enumerate(skeleton):
                pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
                pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))

                if (pos1[0] <= 0 or pos1[0] >= img_w or pos1[1] <= 0
                        or pos1[1] >= img_h or pos2[0] <= 0 or pos2[0] >= img_w
                        or pos2[1] <= 0 or pos2[1] >= img_h
                        or kpts[sk[0], 2] < kpt_score_thr
                        or kpts[sk[1], 2] < kpt_score_thr
                        or pose_link_color[sk_id] is None):
                    # skip the link that should not be drawn
                    continue
                color = tuple(int(c) for c in pose_link_color[sk_id])
                if show_keypoint_weight:
                    img_copy = img.copy()
                    X = (pos1[0], pos2[0])
                    Y = (pos1[1], pos2[1])
                    mX = np.mean(X)
                    mY = np.mean(Y)
                    length = ((Y[0] - Y[1])**2 + (X[0] - X[1])**2)**0.5
                    angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
                    stickwidth = 2
                    polygon = cv2.ellipse2Poly(
                        (int(mX), int(mY)), (int(length / 2), int(stickwidth)),
                        int(angle), 0, 360, 1)
                    cv2.fillConvexPoly(img_copy, polygon, color)
                    transparency = max(
                        0, min(1, 0.5 * (kpts[sk[0], 2] + kpts[sk[1], 2])))
                    cv2.addWeighted(
                        img_copy,
                        transparency,
                        img,
                        1 - transparency,
                        0,
                        dst=img)
                else:
                    cv2.line(img, pos1, pos2, color, thickness=thickness)

    return img