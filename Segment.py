from Libs import * 
from Utils import *


# import segmentor  models
import Segmenters.InternImage.segment
# --------------------------
segmenters = {}
segmenters["InternImage"]    = Segmenters.InternImage.segment
# --------------------------

def segment(args):
    segmenter_name = args.Segmenter
    if segmenter_name not in os.listdir("./Segmenters/"):
        return FailLog(f"{segmenter_name} not recognized in ./Segmenters/")

    cur_segmenter = segmenters[segmenter_name]
    cur_segmenter.segment(args)
    return SucLog("Ran segmentation on video")

def get_results_path_with_frame(results_path, fn):
    splited_path = results_path.split(".")
    return ".".join(splited_path[:-1] + [str(fn)] + splited_path[-1:])

def vis_segment(args):
    video_path = args.Video
    seg_pkl_base = args.SegmentPkl
    annotated_video_path = args.VisSegmentPath
    # tracks_path = args.TrackingPth

    color = np.array([0, 127, 0])
    alpha = 0.35

    cap = cv2.VideoCapture(video_path)
    # Check if camera opened successfully
    if (cap.isOpened()== False): return FailLog("could not open input video")

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    out_cap = cv2.VideoWriter(annotated_video_path,cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width,frame_height))

    if not args.ForNFrames is None:
        frames = int(min(frames, args.ForNFrames))

    # Read until video is completed
    for frame_num in tqdm(range(frames)):
        if (not cap.isOpened()):
            break
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            data_path = get_results_path_with_frame(seg_pkl_base, frame_num)
            df = pd.read_pickle(data_path)
            for i, row in df.iterrows():
                assert row["fn"] ==  frame_num
                x1, y1, x2, y2, raw_mask = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"]), row["mask"]
                mask = raw_mask.astype(bool).reshape(raw_mask.shape[0], raw_mask.shape[1], 1)
                color_mask = mask  * color.reshape(1, 1, -1)
                # color that part of image
                frame[y1:y2, x1:x2][raw_mask] = frame[y1:y2, x1:x2][raw_mask]*( 1- alpha) + color_mask[raw_mask]*alpha

            out_cap.write(frame)
    return SucLog("vis seg executed")