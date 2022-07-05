# Author: Sajjad P. Savaoji April 27 2022
# This py file contains some helper functions for the pipeline

from Libs import *
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Tags:
    WARNING = "[Warning ]"
    FAIL    = "[Error   ]"
    PROC    = "[Proccess]"
    SUCC    = "[Success ]"

class SubTaskExt:
    Detection    = "txt"
    Pkl          = "pkl"
    VisDetection = "MP4"
    Tracking     = "txt"
    VisTracking  = "MP4"
    VisTrajectories = "png"
    VisLTrajectories = "png"
    Json = "json"
    Npy = "npy"

class SubTaskMarker:
    Detection     = "detection"
    VisDetection  = "visdetection"
    Tracking      = "tracking"
    VisTracking   = "vistracking"
    Homography    = "homography"
    VisHomography = "vishomography"
    MetaData      = "metadata"
    VisTrajectories = "vistraj"
    VisLTrajectories = "vislabelledtraj"
    Counting = "counting"
    Clustering = "clustering"
 

class Puncuations:
    Dot = "."

SupportedVideoExts = [".MP4", ".mp4", ".avi", ".AVI"]

class Log(object):
    def __init__(self, message, bcolor, tag) -> None:
        self.message = message
        self.bcolor = bcolor
        self.tag = tag
    def __repr__(self) -> str:
        return f"{self.bcolor}{self.tag}:{bcolors.ENDC} {self.message}"

class WarningLog(Log):
    def __init__(self, message) -> None:
        super().__init__(message, bcolors.WARNING, Tags.WARNING)

class FailLog(Log):
    def __init__(self, message) -> None:
        super().__init__(message, bcolors.FAIL, Tags.FAIL)

class ProcLog(Log):
    def __init__(self, message) -> None:
        super().__init__(message, bcolors.OKCYAN, Tags.PROC)
class SucLog(Log):
    def __init__(self, message) -> None:
        super().__init__(message, bcolors.OKGREEN, Tags.SUCC)

def get_detection_path_from_args(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Detection", file_name + Puncuations.Dot + SubTaskMarker.Detection + Puncuations.Dot + SubTaskExt.Detection)

def get_detection_path_with_detector_from_args(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Detection",file_name + Puncuations.Dot + SubTaskMarker.Detection + Puncuations.Dot + args.Detector + Puncuations.Dot +SubTaskExt.Detection)

def get_detection_pkl(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Detection",file_name + Puncuations.Dot + SubTaskMarker.Detection + Puncuations.Dot + args.Detector + Puncuations.Dot +SubTaskExt.Pkl)

def get_vis_detection_path_from_args(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Visualization",file_name + Puncuations.Dot + SubTaskMarker.VisDetection + Puncuations.Dot + args.Detector + Puncuations.Dot +SubTaskExt.VisDetection)

def get_tracking_path_from_args(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Tracking",file_name + Puncuations.Dot + SubTaskMarker.Tracking + Puncuations.Dot + args.Detector + Puncuations.Dot + args.Tracker + Puncuations.Dot +SubTaskExt.Tracking)

def get_tracking_pkl_from_args(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Tracking",file_name + Puncuations.Dot + SubTaskMarker.Tracking + Puncuations.Dot + args.Detector +Puncuations.Dot + args.Tracker + Puncuations.Dot +SubTaskExt.Pkl)

def get_vis_tracking_path_from_args(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Visualization",file_name + Puncuations.Dot + SubTaskMarker.VisTracking + Puncuations.Dot + args.Detector + Puncuations.Dot + args.Tracker + Puncuations.Dot +SubTaskExt.VisTracking)

def get_plot_all_traj_path(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Visualization",file_name + Puncuations.Dot + SubTaskMarker.VisTrajectories + Puncuations.Dot + args.Detector + Puncuations.Dot + args.Tracker + Puncuations.Dot +SubTaskExt.VisTrajectories)

def get_vis_labelled_tracks_path(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Visualization",file_name + Puncuations.Dot + SubTaskMarker.VisLTrajectories + Puncuations.Dot + args.Detector + Puncuations.Dot + args.Tracker + Puncuations.Dot +SubTaskExt.VisLTrajectories)

def get_homography_streetview_path(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, file_name + Puncuations.Dot + SubTaskMarker.Homography + Puncuations.Dot + "street" + Puncuations.Dot + "png")

def get_metadata_path(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, file_name + Puncuations.Dot + SubTaskMarker.MetaData + Puncuations.Dot + "json")

def get_homography_topview_path(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, file_name + Puncuations.Dot + SubTaskMarker.Homography + Puncuations.Dot + "top" + Puncuations.Dot + "png")

def get_homography_txt_path(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Homography",file_name + Puncuations.Dot + SubTaskMarker.Homography + Puncuations.Dot + "txt")

def get_homography_npy_path(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Homography",file_name + Puncuations.Dot + SubTaskMarker.Homography + Puncuations.Dot + "npy")

def get_homography_csv_path(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Homography",file_name + Puncuations.Dot + SubTaskMarker.Homography + Puncuations.Dot + "csv")

def get_reprojection_path(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Tracking",file_name + Puncuations.Dot + SubTaskMarker.Tracking + Puncuations.Dot + args.Detector + Puncuations.Dot + args.Tracker + Puncuations.Dot + "reprojected" + Puncuations.Dot+SubTaskExt.Tracking)

def get_tracklabelling_export_pth(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Annotation",file_name + Puncuations.Dot + SubTaskMarker.Tracking + Puncuations.Dot + args.Detector + Puncuations.Dot +args.Tracker + Puncuations.Dot + "reprojected" + Puncuations.Dot+ "labelled" + Puncuations.Dot+ SubTaskExt.Pkl)

def get_tracklabelling_export_pth_meter(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Annotation",file_name + Puncuations.Dot + SubTaskMarker.Tracking + Puncuations.Dot + args.Detector + Puncuations.Dot +args.Tracker + Puncuations.Dot + "reprojected" + Puncuations.Dot+ "labelled" + Puncuations.Dot+ "meter" + Puncuations.Dot+SubTaskExt.Pkl)

def get_reprojection_pkl(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Tracking",file_name + Puncuations.Dot + SubTaskMarker.Tracking + Puncuations.Dot + args.Detector+ Puncuations.Dot + args.Tracker + Puncuations.Dot + "reprojected" + Puncuations.Dot+SubTaskExt.Pkl)

def get_reprojection_pkl_meter(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Tracking",file_name + Puncuations.Dot + SubTaskMarker.Tracking + Puncuations.Dot + args.Detector+ Puncuations.Dot + args.Tracker + Puncuations.Dot + "reprojected" + Puncuations.Dot+"meter"+ Puncuations.Dot+SubTaskExt.Pkl)

def get_vishomography_path(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Visualization",file_name + Puncuations.Dot + SubTaskMarker.VisHomography + Puncuations.Dot + "png")

def get_counting_res_pth(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Counting",file_name + Puncuations.Dot + SubTaskMarker.Counting + Puncuations.Dot + args.Detector+ Puncuations.Dot + args.Tracker + Puncuations.Dot +SubTaskExt.Json)

    
def add_detection_pathes_to_args(args):
    d_path = get_detection_path_from_args(args)
    d_d_path = get_detection_path_with_detector_from_args(args)
    d_pkl = get_detection_pkl(args)
    args.DetectionPath = d_path
    args.DetectionDetectorPath = d_d_path
    args.DetectionPkl = d_pkl
    return args

def add_vis_detection_path_to_args(args):
    vis_detection_path = get_vis_detection_path_from_args(args)
    args.VisDetectionPth = vis_detection_path
    return args

def videos_from_dataset(args):
    video_files = []
    all_files = os.listdir(args.Dataset)
    for file in all_files:
        file_name, file_ext = os.path.splitext(file)
        if file_ext in SupportedVideoExts:
            video_files.append(os.path.join(args.Dataset, file))
    return video_files[0]

def add_videos_to_args(args):
    video_file = videos_from_dataset(args)
    args.Video = video_file
    return args

def add_tracking_path_to_args(args):
    tracking_path = get_tracking_path_from_args(args)
    args.TrackingPth = tracking_path
    return args

def add_tracking_pkl_to_args(args):
    tracking_pkl = get_tracking_pkl_from_args(args)
    args.TrackingPkl = tracking_pkl
    return args

def add_vis_tracking_path_to_args(args):
    vis_tracking_pth = get_vis_tracking_path_from_args(args)
    args.VisTrackingPth = vis_tracking_pth
    return args


def add_homographygui_related_path_to_args(args):
    streetview = get_homography_streetview_path(args)
    topview = get_homography_topview_path(args)
    txt = get_homography_txt_path(args)
    npy = get_homography_npy_path(args)
    csv = get_homography_csv_path(args)
    args.HomographyStreetView = streetview
    args.HomographyTopView = topview
    args.HomographyTXT = txt
    args.HomographyNPY = npy
    args.HomographyCSV = csv
    return args

def add_homography_related_path_to_args(args):
    reprojected_path = get_reprojection_path(args)
    reprojected_pkl = get_reprojection_pkl(args)
    args.ReprojectedPoints = reprojected_path
    args.ReprojectedPkl = reprojected_pkl
    return args

def add_vishomography_path_to_args(args):
    vishomographypth = get_vishomography_path(args)
    args.VisHomographyPth = vishomographypth
    return args

def add_tracklabelling_export_to_args(args):
    export_pth = get_tracklabelling_export_pth(args)
    args.TrackLabellingExportPth = export_pth
    return args

def add_metadata_to_args(args):
    meta_path = get_metadata_path(args)
    with open(meta_path) as f:
        args.MetaData = json.load(f)
    return args

def add_plot_all_traj_pth_to_args(args):
    path = get_plot_all_traj_path(args)
    args.PlotAllTrajPth = path
    return args

def add_vis_labelled_tracks_pth_to_args(args):
    path = get_vis_labelled_tracks_path(args)
    args.VisLabelledTracksPth = path
    return args

def add_meter_path_to_args(args):
    reprojected_meter_path = get_reprojection_pkl_meter(args)
    labeled_meter_path = get_tracklabelling_export_pth_meter(args)

    args.ReprojectedPklMeter = reprojected_meter_path
    args.TrackLabellingExportPthMeter = labeled_meter_path
    return args

def add_count_path_to_args(args):
    counting_result_path = get_counting_res_pth(args)
    args.CountingResPth = counting_result_path
    return args

def get_reprojected_meter_cluster_pkl(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Clustering",file_name + Puncuations.Dot + SubTaskMarker.Clustering + Puncuations.Dot + args.ClusteringAlgo + Puncuations.Dot + args.Detector+ Puncuations.Dot + args.Tracker + Puncuations.Dot + "reprojected" + Puncuations.Dot+"meter"+ Puncuations.Dot+SubTaskExt.Pkl)

def get_reprojected_reg_cluster_pkl(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Clustering",file_name + Puncuations.Dot + SubTaskMarker.Clustering + Puncuations.Dot + args.ClusteringAlgo + Puncuations.Dot + args.Detector+ Puncuations.Dot + args.Tracker + Puncuations.Dot + "reprojected" + Puncuations.Dot + Puncuations.Dot+SubTaskExt.Pkl)

def get_distance_matrix_pth(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Clustering",file_name + Puncuations.Dot + SubTaskMarker.Clustering + Puncuations.Dot + args.Detector+ Puncuations.Dot + args.Tracker + Puncuations.Dot + "reprojected"+ Puncuations.Dot+ "DistanceMatrix" + Puncuations.Dot+SubTaskExt.Npy)


def get_vis_clustering_path(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Visualization",file_name + Puncuations.Dot + SubTaskMarker.Clustering + Puncuations.Dot + args.ClusteringAlgo+ Puncuations.Dot + args.Detector+ Puncuations.Dot + args.Tracker + Puncuations.Dot + "reprojected"+ Puncuations.Dot+ "vis" + Puncuations.Dot+"png")


def add_clustering_related_pth_to_args(args):
    meter_clustered = get_reprojected_meter_cluster_pkl(args)
    reg_clustered = get_reprojected_reg_cluster_pkl(args)
    distance_matrix = get_distance_matrix_pth(args)
    vis_path = get_vis_clustering_path(args)

    args.ReprojectedPklMeterCluster = meter_clustered
    args.ReprojectedPklCluster = reg_clustered
    args.ClusteringDistanceMatrix = distance_matrix
    args.ClusteringVis = vis_path
    return args

def complete_args(args):
    if args.Video is None:
        # if Video path was not specified by the user grab a video from dataset
        args = add_videos_to_args(args)

    if (not args.Detector is None) or args.DetPostProc:
        args = add_detection_pathes_to_args(args)
        args = add_vis_detection_path_to_args(args)

    if not args.Tracker is None:
        args = add_tracking_path_to_args(args)
        args = add_vis_tracking_path_to_args(args)
        args = add_tracking_pkl_to_args(args)

    args = add_metadata_to_args(args)
    if args.HomographyGUI or args.Homography or args.VisHomographyGUI or args.VisTrajectories or args.VisLabelledTrajectories or args.Cluster:
        args = add_homographygui_related_path_to_args(args)
    if args.Homography or args.VisTrajectories or args.VisLabelledTrajectories or args.Meter or args.Cluster:
        args = add_homography_related_path_to_args(args)
    if args.VisHomographyGUI or args.VisLabelledTrajectories:
        args = add_vishomography_path_to_args(args)
    if args.TrackLabelingGUI or args.VisLabelledTrajectories or args.Meter:
        args = add_tracklabelling_export_to_args(args)
    if args.VisTrajectories:
        args = add_plot_all_traj_pth_to_args(args)
    if args.VisLabelledTrajectories:
        args = add_vis_labelled_tracks_pth_to_args(args)
    if args.Meter or args.Count or args.Cluster:
        args = add_meter_path_to_args(args)
    if args.Count:
        args = add_count_path_to_args(args)
    if args.Cluster:
        args = add_clustering_related_pth_to_args(args)

    return args

def check_config(args):
    # check if args passed are valid
    # create Results folders Accordigly
    # check if Resutls folder is in the dataset location
    results_path= os.path.join(args.Dataset, "Results")
    detection_path = os.path.join(results_path, "Detection")
    tracking_path = os.path.join(results_path, "Tracking")
    homography_path = os.path.join(results_path, "Homography")
    Vis_path = os.path.join(results_path, "Visualization")
    Annotation_path = os.path.join(results_path, "Annotation")
    counting_path = os.path.join(results_path, "Counting")
    clustering_path = os.path.join(results_path, "Clustering")

    try: os.system(f"mkdir -p {results_path}")
    except: pass
    try: os.system(f"mkdir -p {detection_path}")
    except: pass
    try: os.system(f"mkdir -p {tracking_path}")
    except: pass
    try: os.system(f"mkdir -p {homography_path}")
    except: pass
    try: os.system(f"mkdir -p {Vis_path}")
    except: pass
    try: os.system(f"mkdir -p {counting_path}")
    except: pass
    try: os.system(f"mkdir -p {Annotation_path}")
    except: pass
    try: os.system(f"mkdir -p {clustering_path}")
    except: pass

def get_conda_envs():
    stream = os.popen("conda env list")
    output = stream.read()
    a=output.split()
    a.remove("*")
    return output.split()[4::2]

def make_conda_env(env_name, libs=""):
    os.system(f"conda create -n {env_name} -y "+libs)

def activate_conda_env(env_name):
    os.system(f"conda activate {env_name}")

def deactivate_conda_env(env_name):
    os.system(f"conda deactivate")

def conda_pyrun(env_name, exec_file, args):
    os.system(f"conda run -n {env_name} python3 \"{exec_file}\" '{json.dumps(dict(vars(args)))}'")

def download_url_to(url, path):
    # make sure that path is valid
    r = requests.get(url, allow_redirects=True)
    open(path, 'wb').write(r.content)

def save_frame_from_video(video_path, image_out_path):
    chosen_frame = 1 # leave it this way for now
    # Opens the Video file
    cap = cv2.VideoCapture(video_path)
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    for frame_num in tqdm(range(frames)):
        if (not cap.isOpened()):
            break
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret and (frame_num == chosen_frame):
            cv2.imwrite(image_out_path,frame)
            break
    cap.release()
    cv2.destroyAllWindows()