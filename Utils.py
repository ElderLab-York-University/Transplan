# Author: Sajjad P. Savaoji April 27 2022
# This py file contains some helper functions for the pipeline
from Libs import *
moi_color_dict = {
    0:(0, 0, 0),
    1:(128, 0, 0),
    2:(230, 25, 75),
    3:(250, 190, 212),
    4:(170, 110, 40),
    5:(245, 130, 48),
    6:(255, 215, 180),
    7:(128, 128, 0),
    8:(255, 255, 25),
    9:(210, 245, 60),
    10:(0, 0, 128),
    11:(70, 240, 240),
    12:(0, 130, 200),
}


class_color_dict={
    0:(0, 255,255),
    1:(0,255,0),
    2:(230, 0, 0),
    # 3:(250, 190, 212),
    # 4:(170, 110, 40),
    5:(255,255,255),
    # 6:(255, 215, 180),
    7:(0, 0, 255),
    # 8:(255, 255, 25),
    # 9:(210, 245, 60),
    # 10:(0, 0, 128),
    # 11:(70, 240, 240),
    # 12:(0, 130, 200),    
}
roi_color_dict = {
    0:(0, 0, 0),
    1:(128, 0, 0),
    2:(170, 110, 40),
    3:(128, 128, 0),
    4:(0, 0, 128),
    5:(0, 0, 0),
    6:(0, 0, 0),
    7:(0, 0, 0),

}
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
    Csv = "csv"

class SubTaskMarker:
    Detection     = "detection"
    VisDetection  = "visdetection"
    Tracking      = "tracking"
    VisTracking   = "vistracking"
    VisMoITracking = "visMOItracking"
    Homography    = "homography"
    VisHomography = "vishomography"
    MetaData      = "metadata"
    VisTrajectories = "vistraj"
    VisLTrajectories = "vislabelledtraj"
    Counting = "counting"
    Clustering = "clustering"
    VisROI = "visROI"
    IdMatched = "IdMatched"
    MCTrackDis = "MCTrackDist"
    
class Puncuations:
    Dot = "."

SupportedVideoExts = [".MP4", ".mp4", ".avi", ".AVI", ".mkv", ".MKV"]

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
def add_gt_mask_path_to_args(args):
    args.GTMask= get_gt_mask_path(args)
    return args
def replace_white_space(s):
    return s.replace(" ",",")
def convert_pandas_versions(args, df_to_convert):
    args.df_to_convert=df_to_convert
    args.temp_text_path="./temp_df.txt"
    args.temp_cols_path='./temp_cols.txt'
    conda_pyrun("pandasver", "convert_df_to_txt.py", args)
    df=pd.read_csv(args.temp_text_path, index_col=0, sep=",")
    df.set_index(df.iloc[:, 0])
    for column in df.columns:
        if( "True"  in str(df[column]) or "False" in str(df[column])):
            pattern1= re.compile(r'(?<=[a-zA-Z])\s+(?=[a-zA-Z])')
            pattern2=re.compile(r'[\n\\n]')
            print(column)
            df[column]=df[column].apply(lambda x:re.sub(pattern2,',',re.sub(pattern1,',',x)))
            df[column]=df[column].apply(ast.literal_eval)
        else:
            try:
                df[column]=df[column].apply(ast.literal_eval)
                
            except SyntaxError as e:
                pattern = re.compile(r'(\d)\s+(\d)')
                pattern2 = re.compile(r'(\d+)\.(?![0-9])')       
                print(df[column])
                    
                df[column]=df[column].apply(lambda x:re.sub(pattern,r'\1,\2',re.sub(pattern2,r'\1.0',x)).replace("\n", ','))
                print(df[column])
                
                df[column]=df[column].apply(ast.literal_eval)
            except ValueError as e:
                print(e)
                pass
    # for index, value in df["trajectory"].iteritems():
    #     print(f"Row {index}: trajectory - {type(value)}") 
    
    print(df)  
    pd.to_pickle(df, getattr(args, df_to_convert))
    
def get_detection_path_from_args(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Detection", file_name + Puncuations.Dot + SubTaskMarker.Detection + Puncuations.Dot + SubTaskExt.Detection)

def get_detection_path_with_detector_from_args(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Detection",file_name + Puncuations.Dot + SubTaskMarker.Detection + Puncuations.Dot + args.Detector + Puncuations.Dot +SubTaskExt.Detection)
def get_detection_roi_path_with_detector_from_args(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Detection",file_name + Puncuations.Dot + SubTaskMarker.Detection + Puncuations.Dot + args.Detector+ Puncuations.Dot+ args.roi_num + Puncuations.Dot +SubTaskExt.Detection)

def get_roi_npz_path_from_args(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    npz_path = os.path.join(args.Dataset, file_name+Puncuations.Dot+"roi"+Puncuations.Dot+args.roi_num+Puncuations.Dot+"npz")
    return npz_path
def get_detection_pkl(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Detection",file_name + Puncuations.Dot + SubTaskMarker.Detection + Puncuations.Dot + args.Detector + Puncuations.Dot +SubTaskExt.Pkl)
def get_detection_roi_pkl(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Detection",file_name + Puncuations.Dot + SubTaskMarker.Detection + Puncuations.Dot + args.Detector + Puncuations.Dot+ args.roi_num + Puncuations.Dot  +SubTaskExt.Pkl)

def get_detection_coco(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Detection",file_name + Puncuations.Dot + SubTaskMarker.Detection + Puncuations.Dot + args.Detector + Puncuations.Dot+ "COCO"+ Puncuations.Dot+SubTaskExt.Json)

def get_detection_pkl_back_up(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Detection",file_name + Puncuations.Dot + SubTaskMarker.Detection + Puncuations.Dot + args.Detector + Puncuations.Dot + "backup" + Puncuations.Dot +SubTaskExt.Pkl)
def get_detection_roi_pkl_back_up(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Detection",file_name + Puncuations.Dot + SubTaskMarker.Detection + Puncuations.Dot + args.Detector + Puncuations.Dot + "backup" + Puncuations.Dot+ args.roi_num+Puncuations.Dot +SubTaskExt.Pkl)

def get_vis_detection_path_from_args(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Visualization",file_name + Puncuations.Dot + SubTaskMarker.VisDetection + Puncuations.Dot + args.Detector + Puncuations.Dot +SubTaskExt.VisDetection)
def get_vis_detection_roi_path_from_args(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Visualization",file_name + Puncuations.Dot + SubTaskMarker.VisDetection + Puncuations.Dot + args.Detector+Puncuations.Dot+args.roi_num + Puncuations.Dot +SubTaskExt.VisDetection)
    
def get_tracking_path_from_args(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Tracking",file_name + Puncuations.Dot + SubTaskMarker.Tracking + Puncuations.Dot + args.Detector + Puncuations.Dot + args.Tracker + Puncuations.Dot +SubTaskExt.Tracking)

def get_tracking_roi_path_from_args(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Tracking",file_name + Puncuations.Dot + SubTaskMarker.Tracking + Puncuations.Dot + args.Detector+Puncuations.Dot+args.roi_num + Puncuations.Dot + args.Tracker + Puncuations.Dot +SubTaskExt.Tracking)

def get_tracking_pkl_from_args(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Tracking",file_name + Puncuations.Dot + SubTaskMarker.Tracking + Puncuations.Dot + args.Detector +Puncuations.Dot + args.Tracker + Puncuations.Dot +SubTaskExt.Pkl)

def get_tracking_roi_pkl_from_args(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Tracking",file_name + Puncuations.Dot + SubTaskMarker.Tracking + Puncuations.Dot + args.Detector + args.Detector+Puncuations.Dot+args.roi_num +Puncuations.Dot + args.Tracker + Puncuations.Dot +SubTaskExt.Pkl)

def get_tracking_pkl_bu_from_args(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Tracking",file_name + Puncuations.Dot + SubTaskMarker.Tracking + Puncuations.Dot + args.Detector +Puncuations.Dot + args.Tracker + Puncuations.Dot +"back" + Puncuations.Dot +SubTaskExt.Pkl)
def get_tracking_roi_pkl_bu_from_args(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Tracking",file_name + Puncuations.Dot + SubTaskMarker.Tracking + Puncuations.Dot + args.Detector + args.Detector+Puncuations.Dot+args.roi_num +Puncuations.Dot + args.Tracker + Puncuations.Dot +"back" + Puncuations.Dot +SubTaskExt.Pkl)

def get_vis_tracking_path_from_args(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Visualization",file_name + Puncuations.Dot + SubTaskMarker.VisTracking + Puncuations.Dot + args.Detector + Puncuations.Dot + args.Tracker + Puncuations.Dot +SubTaskExt.VisTracking)
def get_vis_tracking_roi_path_from_args(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Visualization",file_name + Puncuations.Dot + SubTaskMarker.VisTracking + Puncuations.Dot + args.Detector+Puncuations.Dot+args.roi_num  + Puncuations.Dot + args.Tracker + Puncuations.Dot +SubTaskExt.VisTracking)


def get_vis_top_tracking_path_from_args(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Visualization",file_name + Puncuations.Dot + "VisTrackingTop" + Puncuations.Dot + args.Detector + Puncuations.Dot + args.Tracker + Puncuations.Dot +SubTaskExt.VisTracking)

def get_vis_tracking_moi_path_from_args(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Visualization",file_name + Puncuations.Dot + SubTaskMarker.VisMoITracking + Puncuations.Dot + args.Detector + Puncuations.Dot + args.Tracker + Puncuations.Dot + args.CountMetric +Puncuations.Dot+ SubTaskExt.VisTracking)

def get_plot_all_traj_path(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Visualization",file_name + Puncuations.Dot + SubTaskMarker.VisTrajectories + Puncuations.Dot + args.Detector + Puncuations.Dot + args.Tracker + Puncuations.Dot +SubTaskExt.VisTrajectories)

def get_vis_labelled_tracks_path(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Visualization",file_name + Puncuations.Dot + SubTaskMarker.VisLTrajectories + Puncuations.Dot + args.Detector + Puncuations.Dot + args.Tracker + Puncuations.Dot +SubTaskExt.VisLTrajectories)

def get_vis_labelled_tracks_path_image(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Visualization",file_name + Puncuations.Dot + SubTaskMarker.VisLTrajectories + Puncuations.Dot +"Street"+Puncuations.Dot+args.Detector + Puncuations.Dot + args.Tracker + Puncuations.Dot +SubTaskExt.VisLTrajectories)

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
    return os.path.join(args.Dataset, file_name + Puncuations.Dot + SubTaskMarker.Homography + Puncuations.Dot + "top"+ Puncuations.Dot + args.TopView + Puncuations.Dot + "png")

def get_homography_txt_path(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Homography",file_name + Puncuations.Dot + SubTaskMarker.Homography + Puncuations.Dot + args.TopView + Puncuations.Dot + "txt")

def get_homography_npy_path(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Homography",file_name + Puncuations.Dot + SubTaskMarker.Homography + Puncuations.Dot + args.TopView + Puncuations.Dot + "npy")

def get_homography_csv_path(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Homography",file_name + Puncuations.Dot + SubTaskMarker.Homography + Puncuations.Dot + args.TopView + Puncuations.Dot + "csv")

def get_reprojection_path(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Tracking",file_name + Puncuations.Dot + SubTaskMarker.Tracking + Puncuations.Dot + args.Detector + Puncuations.Dot + args.Tracker + Puncuations.Dot + "reprojected" + Puncuations.Dot+ args.BackprojectionMethod + Puncuations.Dot + args.TopView + Puncuations.Dot + args.ContactPoint+ Puncuations.Dot +SubTaskExt.Tracking)

def get_reprojection_path_for_detection(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Detection",file_name + Puncuations.Dot + SubTaskMarker.Detection + Puncuations.Dot + args.Detector + Puncuations.Dot + "reprojected" + Puncuations.Dot+ args.BackprojectionMethod+ Puncuations.Dot+ args.TopView + Puncuations.Dot+ args.ContactPoint+ Puncuations.Dot+ SubTaskExt.Detection)

def get_tracklabelling_export_pth(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Annotation",file_name + Puncuations.Dot + SubTaskMarker.Tracking + Puncuations.Dot + args.Detector + Puncuations.Dot +args.Tracker + Puncuations.Dot + "reprojected" + Puncuations.Dot+ "labelled" + Puncuations.Dot+ SubTaskExt.Pkl)

def get_tracklabelling_export_pth_image(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Annotation",file_name + Puncuations.Dot + SubTaskMarker.Tracking + Puncuations.Dot + args.Detector + Puncuations.Dot +args.Tracker + Puncuations.Dot + "reprojected" + Puncuations.Dot+ "labelled" + Puncuations.Dot+"Image"+Puncuations.Dot+ SubTaskExt.Pkl)

def get_tracklabelling_export_pth_meter(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Annotation",file_name + Puncuations.Dot + SubTaskMarker.Tracking + Puncuations.Dot + args.Detector + Puncuations.Dot +args.Tracker + Puncuations.Dot + "reprojected" + Puncuations.Dot+ "labelled" + Puncuations.Dot+ "meter" + Puncuations.Dot+SubTaskExt.Pkl)

def get_reprojection_pkl(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Tracking",file_name + Puncuations.Dot + SubTaskMarker.Tracking + Puncuations.Dot + args.Detector+ Puncuations.Dot + args.Tracker + Puncuations.Dot + "reprojected" + Puncuations.Dot+ args.BackprojectionMethod + Puncuations.Dot+ args.TopView + Puncuations.Dot+ args.ContactPoint+args.TopView +SubTaskExt.Pkl)

def get_reprojection_pkl_for_detection(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Detection",file_name + Puncuations.Dot + SubTaskMarker.Detection + Puncuations.Dot + args.Detector+ Puncuations.Dot + "reprojected" + Puncuations.Dot+ args.BackprojectionMethod + Puncuations.Dot+ args.TopView + Puncuations.Dot+ args.ContactPoint+ Puncuations.Dot+SubTaskExt.Pkl)

def get_vis_cp_path_for_detection(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Visualization",file_name + Puncuations.Dot + "VisContactPoints" + Puncuations.Dot + args.Detector+ Puncuations.Dot+ args.BackprojectionMethod + Puncuations.Dot+ args.TopView + Puncuations.Dot+ args.ContactPoint+Puncuations.Dot+SubTaskExt.VisTracking)

def get_vis_cp_top_for_detection(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Visualization",file_name + Puncuations.Dot + "VisCPTop" + Puncuations.Dot + args.Detector+ Puncuations.Dot+ args.BackprojectionMethod + Puncuations.Dot+ args.TopView + Puncuations.Dot+ args.ContactPoint+Puncuations.Dot+SubTaskExt.VisTracking)

def get_reprojection_pkl_meter(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Tracking",file_name + Puncuations.Dot + SubTaskMarker.Tracking + Puncuations.Dot + args.Detector+ Puncuations.Dot + args.Tracker + Puncuations.Dot + "reprojected" + Puncuations.Dot+"meter"+ Puncuations.Dot+SubTaskExt.Pkl)

def get_vishomography_path(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Visualization",file_name + Puncuations.Dot + SubTaskMarker.VisHomography + Puncuations.Dot + "png")

def get_vis_roi_path(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Visualization",file_name + Puncuations.Dot + SubTaskMarker.VisROI + Puncuations.Dot + "png")

def get_counting_res_pth(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Counting",file_name + Puncuations.Dot + SubTaskMarker.Counting + Puncuations.Dot + args.Detector+ Puncuations.Dot + args.Tracker + Puncuations.Dot + args.CountMetric +Puncuations.Dot +SubTaskExt.Json)

def get_vis_density_path(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Visualization",file_name + Puncuations.Dot + "density" + Puncuations.Dot + args.Detector+ Puncuations.Dot + args.Tracker + Puncuations.Dot + args.CountMetric +Puncuations.Dot)

def get_counting_stat_pth(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Counting",file_name + Puncuations.Dot + SubTaskMarker.Counting + Puncuations.Dot + args.Detector+ Puncuations.Dot + args.Tracker + Puncuations.Dot + args.CountMetric +Puncuations.Dot +SubTaskExt.Csv)

def get_counter_cached_path(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Counting",file_name + Puncuations.Dot + SubTaskMarker.Counting + Puncuations.Dot + args.Detector+ Puncuations.Dot + args.Tracker + Puncuations.Dot + args.CountMetric +Puncuations.Dot + "cached" +Puncuations.Dot +SubTaskExt.Pkl)

def get_counting_idmatching_pth(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Counting",file_name + Puncuations.Dot + SubTaskMarker.IdMatched + Puncuations.Dot + args.Detector+ Puncuations.Dot + args.Tracker + Puncuations.Dot + args.CountMetric +Puncuations.Dot +SubTaskExt.Csv)

def get_dsm_points_path(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, file_name + Puncuations.Dot + "dsm_points_aoi" + Puncuations.Dot + "xyz")

def get_intrinsics_path(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, file_name + Puncuations.Dot + "intrinsic_calibration" + Puncuations.Dot + "json")

def get_extrinsics_path(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, file_name + Puncuations.Dot + "extrinsic_calibration" + Puncuations.Dot + "json")

def get_orthophoto_tif_path(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, file_name + Puncuations.Dot + "OrthoPhoto" + Puncuations.Dot + "tif")

def get_to_ground_correspondance_path(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/DSM/" + Puncuations.Dot + "ToGroundCorrespondance" + Puncuations.Dot + "csv")

def get_to_ground_raster_path(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/DSM/" + Puncuations.Dot + "ToGroundRaster" + Puncuations.Dot + "pkl")

def add_detection_pathes_to_args(args):
    d_path = get_detection_path_from_args(args)
    d_d_path = get_detection_path_with_detector_from_args(args)
    d_pkl = get_detection_pkl(args)
    d_pkl_bu = get_detection_pkl_back_up(args)
    args.DetectionPath = d_path
    args.DetectionDetectorPath = d_d_path
    args.DetectionPkl = d_pkl
    args.DetectionPklBackUp = d_pkl_bu
    args.DetectionCOCO = get_detection_coco(args)
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
    tracking_pkl_bu = get_tracking_pkl_bu_from_args(args)
    args.TrackingPkl = tracking_pkl
    args.TrackingPklBackUp = tracking_pkl_bu
    return args

def add_vis_tracking_path_to_args(args):
    vis_tracking_pth = get_vis_tracking_path_from_args(args)
    args.VisTrackingPth = vis_tracking_pth
    return args

def add_vis_top_tracking_path_to_args(args):
    vis_top_tracking_path  = get_vis_top_tracking_path_from_args(args)
    args.VisTrackingTopPth = vis_top_tracking_path
    return args

def add_vis_tracking_moi_path_to_args(args):
    vis_tracking_pth = get_vis_tracking_moi_path_from_args(args)
    args.VisTrackingMoIPth = vis_tracking_pth
    return args

def get_track_eval_save_path(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Tracking",file_name + Puncuations.Dot + "EvaluateTracking" + Puncuations.Dot + args.GTDetector+ Puncuations.Dot + args.GTTracker + Puncuations.Dot+ args.Detector+ Puncuations.Dot + args.Tracker + Puncuations.Dot+"txt")
def get_detect_eval_save_path(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    if(args.Rois is not None):
        roistr="".join([str(args.Rois[i]+args.Rois[i+1]) for i in range(0, len(args.Rois),2)])+ Puncuations.Dot
    else:
        roistr=""
    if(args.Camera is None):
        return os.path.join(args.Dataset, "Results/Detection",file_name + Puncuations.Dot + "EvaluateDetection" + Puncuations.Dot + args.GTDetector+ Puncuations.Dot + args.Detector+ Puncuations.Dot+roistr+"txt")
    else:
        return os.path.join(args.Dataset, "Results/Detection",file_name + Puncuations.Dot + "EvaluateDetection" + Puncuations.Dot + args.GTDetector+ Puncuations.Dot + args.Detector+Puncuations.Dot+ "".join((args.Camera)) + Puncuations.Dot+roistr+"txt")
def get_detect_eval_Intersections_save_path(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    if(args.Rois is not None):
        roistr="".join([str(args.Rois[i]+args.Rois[i+1]) for i in range(0, len(args.Rois),2)])+ Puncuations.Dot
    else:
        roistr=""
    
    if(args.Camera is None):
    
        return os.path.join(args.Dataset, "Results/Detection",file_name + Puncuations.Dot + "EvaluateDetection" + Puncuations.Dot + args.GTDetector+ Puncuations.Dot + args.Detector+Puncuations.Dot+"Intersection"+ Puncuations.Dot+roistr+"txt")
    else:
        return os.path.join(args.Dataset, "Results/Detection",file_name + Puncuations.Dot + "EvaluateDetection" + Puncuations.Dot + args.GTDetector+ Puncuations.Dot + args.Detector+Puncuations.Dot+"Intersection"+Puncuations.Dot+ "".join((args.Camera)) + roistr+Puncuations.Dot+"txt")
        
def get_detect_eval_NonIntersections_save_path(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    if(args.Rois is not None):
        roistr="".join([str(args.Rois[i]+args.Rois[i+1]) for i in range(0, len(args.Rois),2)])+ Puncuations.Dot
    else:
        roistr=""
    
    if(args.Camera is None):
    
        return os.path.join(args.Dataset, "Results/Detection",file_name + Puncuations.Dot + "EvaluateDetection" + Puncuations.Dot + args.GTDetector+ Puncuations.Dot + args.Detector+Puncuations.Dot+"NonIntersection"+ Puncuations.Dot+ roistr+"txt")
    else:
        return os.path.join(args.Dataset, "Results/Detection",file_name + Puncuations.Dot + "EvaluateDetection" + Puncuations.Dot + args.GTDetector+ Puncuations.Dot + args.Detector+Puncuations.Dot+"NonIntersection"+ Puncuations.Dot+ "".join((args.Camera))+ Puncuations.Dot+ roistr+"txt")
        
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
    if args.Tracker is not None:
        reprojected_path = get_reprojection_path(args)
        reprojected_pkl = get_reprojection_pkl(args)
        args.ReprojectedPoints = reprojected_path
        args.ReprojectedPkl = reprojected_pkl
    if args.Detector is not None:
        args.ReprojectedPointsForDetection = get_reprojection_path_for_detection(args)
        args.ReprojectedPklForDetection = get_reprojection_pkl_for_detection(args)
        args.VisContactPointPth = get_vis_cp_path_for_detection(args)
        args.VisContactPointTopPth = get_vis_cp_top_for_detection(args)

    return args

def add_calibration_related_path_to_args(args):
    args.INTRINSICS_PATH = get_intrinsics_path(args)
    args.EXTRINSICS_PATH = get_extrinsics_path(args)
    return args

def add_dsm_related_path_to_args(args):
    args.DSM_POINTS_PATH        = get_dsm_points_path(args)
    args.ToGroundCorrespondance = get_to_ground_correspondance_path(args)
    args.ToGroundRaster         = get_to_ground_raster_path(args)
    args.OrthoPhotoTif          = get_orthophoto_tif_path(args)
    return args

def add_vishomography_path_to_args(args):
    vishomographypth = get_vishomography_path(args)
    args.VisHomographyPth = vishomographypth
    return args

def add_visroi_path_to_args(args):
    vis_roi_path = get_vis_roi_path(args)
    args.VisROIPth = vis_roi_path
    return args

def add_tracklabelling_export_to_args(args):
    export_pth = get_tracklabelling_export_pth(args)
    export_pth_image = get_tracklabelling_export_pth_image(args)
    args.TrackLabellingExportPth = export_pth
    args.TrackLabellingExportPthImage = export_pth_image
    return args

def add_metadata_to_args(args):
    meta_path = get_metadata_path(args)
    with open(meta_path) as f:
        args.MetaData = json.load(f)
    return args

def add_correct_roi_to_args(args):
    import Homography
    if not args.ROIFromTop: return args
    M, GroundRaster, orthophoto_win_tif_obj = get_projection_objs(args)
    if args.TopView == "GoogleMap":
        ref_roi = args.MetaData["GoogleMapROI"]
    elif args.TopView == "OrthoPhoto":
        ref_roi = args.MetaData["OrthoPhotoROI"]
    else: raise NotImplementedError

    projected_roi = []
    for x, y in ref_roi:
        cor_img = Homography.project_point(args, x, y, args.BackprojectionMethod,
                                M=M, GroundRaster=GroundRaster, TifObj = orthophoto_win_tif_obj)
        x_img, y_img = int(cor_img[0]), int(cor_img[1])
        projected_roi.append([x_img, y_img])
    
    # overwrite roi
    args.MetaData["roi"] = projected_roi
    return args

def get_projection_objs(args):
    import DSM
    method = args.BackprojectionMethod
    M = None
    GroundRaster = None
    orthophoto_win_tif_obj = None

    if method == "Homography":
        homography_path = args.HomographyNPY
        M = np.load(homography_path, allow_pickle=True)[0]

    elif method == "DSM":
        # load OrthophotoTiffile
        orthophoto_win_tif_obj, __ = DSM.load_orthophoto(args.OrthoPhotoTif)
        # creat/load raster
        if not os.path.exists(args.ToGroundRaster):
            coords = DSM.load_dsm_points(args)
            intrinsic_dict = DSM.load_json_file(args.INTRINSICS_PATH)
            projection_dict = DSM.load_json_file(args.EXTRINSICS_PATH)
            DSM.create_raster(args, args.MetaData["camera"], coords, projection_dict, intrinsic_dict)
        
        with open(args.ToGroundRaster, 'rb') as f:
            GroundRaster = DSM.pickle.load(f)
    return M, GroundRaster, orthophoto_win_tif_obj

def add_plot_all_traj_pth_to_args(args):
    path = get_plot_all_traj_path(args)
    args.PlotAllTrajPth = path
    return args

def add_track_eval_save_path(args):
    args.TrackEvalPth = get_track_eval_save_path(args)
    return args 

def add_detect_eval_save_path(args):
    if not args.Intersection and not args.NotIntersection:
        args.DetectEvalPth = get_detect_eval_save_path(args)
    elif args.Intersection:
        args.DetectEvalPth=get_detect_eval_Intersections_save_path(args)
    else:
        args.DetectEvalPth=get_detect_eval_NonIntersections_save_path(args)
    return args 

def add_vis_labelled_tracks_pth_to_args(args):
    path = get_vis_labelled_tracks_path(args)
    path_image = get_vis_labelled_tracks_path_image(args)
    args.VisLabelledTracksPth = path
    args.VisLabelledTracksPthImage = path_image
    return args

def add_meter_path_to_args(args):
    reprojected_meter_path = get_reprojection_pkl_meter(args)
    labeled_meter_path = get_tracklabelling_export_pth_meter(args)

    args.ReprojectedPklMeter = reprojected_meter_path
    args.TrackLabellingExportPthMeter = labeled_meter_path
    return args

def add_count_path_to_args(args):
    counting_result_path = get_counting_res_pth(args)
    counting_stat_path = get_counting_stat_pth(args)
    counting_idmatching_path = get_counting_idmatching_pth(args)
    args.CountingResPth = counting_result_path
    args.CountingStatPth = counting_stat_path
    args.CountingIdMatchPth = counting_idmatching_path
    return args

def get_reprojected_meter_cluster_pkl(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Clustering",file_name + Puncuations.Dot + SubTaskMarker.Clustering + Puncuations.Dot + args.ClusteringAlgo + Puncuations.Dot + args.ClusterMetric + Puncuations.Dot + args.Detector+ Puncuations.Dot + args.Tracker + Puncuations.Dot + "reprojected" + Puncuations.Dot+"meter"+ Puncuations.Dot+SubTaskExt.Pkl)

def get_reprojected_reg_cluster_pkl(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Clustering",file_name + Puncuations.Dot + SubTaskMarker.Clustering + Puncuations.Dot + args.ClusteringAlgo + Puncuations.Dot + args.ClusterMetric + Puncuations.Dot + args.Detector+ Puncuations.Dot + args.Tracker + Puncuations.Dot + "reprojected" + Puncuations.Dot + Puncuations.Dot+SubTaskExt.Pkl)

def get_distance_matrix_pth(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Clustering",file_name + Puncuations.Dot + SubTaskMarker.Clustering + Puncuations.Dot + args.ClusterMetric + Puncuations.Dot + args.Detector+ Puncuations.Dot + args.Tracker + Puncuations.Dot + "reprojected"+ Puncuations.Dot+ "DistanceMatrix" + Puncuations.Dot+SubTaskExt.Npy)


def get_vis_clustering_path(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Visualization",file_name + Puncuations.Dot + SubTaskMarker.Clustering + Puncuations.Dot + args.ClusteringAlgo+ Puncuations.Dot + args.ClusterMetric + Puncuations.Dot + args.Detector+ Puncuations.Dot + args.Tracker + Puncuations.Dot + "reprojected"+ Puncuations.Dot+ "vis" + Puncuations.Dot+"png")


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

def add_cached_counter_path_to_args(args):
    if args.CachedCounterPth is None:
        cached_path = get_counter_cached_path(args)
        args.CachedCounterPth = cached_path
    elif args.UseCachedCounter:
        import main
        parser = main.get_parser()
        chached_args = parser.parse_args()
        chached_args.Dataset = args.CachedCounterPth
        chached_args.CachedCounterPth = None

        if args.MultiCam:
            chached_args, _       = get_args_mc(chached_args)
        elif args.MultiSeg:
            chached_args, _, _    = get_args_ms(chached_args)
        elif args.MultiPart:
            chached_args, _, _, _ = get_args_mp(chached_args)
        else:
            chached_args = get_args(chached_args)

        args.CachedCounterPth = chached_args.CachedCounterPth

    return args

def add_density_path_to_args(args):
    vis_density_pth = get_vis_density_path(args)
    args.DensityVisPath = vis_density_pth
    return args

def get_segmentation_pkl_path(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Segment",file_name + Puncuations.Dot + "Segmentation" + Puncuations.Dot + args.Segmenter + Puncuations.Dot + SubTaskExt.Pkl )

def get_segmentation_pkl_path_backup(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Segment",file_name + Puncuations.Dot + "Segmentation" + Puncuations.Dot + args.Segmenter + Puncuations.Dot+ "BackUp"+ Puncuations.Dot + SubTaskExt.Pkl )

def get_vis_segment_path(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Visualization",file_name + Puncuations.Dot + "VisSegment" + Puncuations.Dot + args.Segmenter + Puncuations.Dot +SubTaskExt.VisTracking)

def add_segmentation_path_to_args(args):
    args.SegmentPkl = get_segmentation_pkl_path(args)
    args.SegmentPklBackUp = get_segmentation_pkl_path_backup(args)
    return args

def add_vis_segment_path_to_args(args):
    args.VisSegmentPath = get_vis_segment_path(args)
    return args

def get_opt_bw_image(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Visualization",file_name + Puncuations.Dot + "OptBW.Image" + Puncuations.Dot + args.Detector + Puncuations.Dot + args.Tracker + Puncuations.Dot +f"{args.OSR}"+ Puncuations.Dot +f"{args.ResampleTH}" +Puncuations.Dot + "png")

def get_opt_bw_ground(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/Visualization",file_name + Puncuations.Dot + "OptBW.GP" + Puncuations.Dot + args.Detector + Puncuations.Dot + args.Tracker + Puncuations.Dot +f"{args.OSR}"+ Puncuations.Dot +f"{args.ResampleTH}"+ Puncuations.Dot+ args.BackprojectionMethod + Puncuations.Dot + args.TopView + Puncuations.Dot + args.ContactPoint+ Puncuations.Dot +"png")
def get_calculate_distance_vis_pth(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    roi_str="" if args.Rois is None else Puncuations.Dot+ str(args.roi_num)
    
    return os.path.join(args.Dataset, "Results/Visualization",file_name + Puncuations.Dot + "VisDistance" + Puncuations.Dot+args.BackprojectionMethod +Puncuations.Dot + args.Detector+roi_str+ Puncuations.Dot + args.Tracker + Puncuations.Dot +SubTaskExt.VisTracking)
def get_calculate_speed_vis_pth(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    roi_str="" if args.Rois is None else Puncuations.Dot+ str(args.roi_num)
    
    return os.path.join(args.Dataset, "Results/Visualization",file_name + Puncuations.Dot + "VisSpeed" + Puncuations.Dot+args.BackprojectionMethod +Puncuations.Dot + args.Detector+roi_str+ Puncuations.Dot + args.Tracker + Puncuations.Dot +SubTaskExt.VisTracking)
def get_vis_all_path(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    roi_str="" if args.Rois is None else Puncuations.Dot+ str(args.roi_num)
    
    return os.path.join(args.Dataset, "Results/Visualization",file_name + Puncuations.Dot + "VisAll" + Puncuations.Dot+args.BackprojectionMethod +Puncuations.Dot + args.Detector+roi_str+ Puncuations.Dot + args.Tracker + Puncuations.Dot +SubTaskExt.VisTracking)
def get_vis_class_path(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    roi_str="" if args.Rois is None else Puncuations.Dot+ str(args.roi_num)
    
    return os.path.join(args.Dataset, "Results/Visualization",file_name + Puncuations.Dot + "VisClass" + Puncuations.Dot+args.BackprojectionMethod +Puncuations.Dot + args.Detector+roi_str+ Puncuations.Dot + args.Tracker + Puncuations.Dot +SubTaskExt.VisTracking)
def get_vis_lanes_path(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    roi_str="" if args.Rois is None else Puncuations.Dot+ str(args.roi_num)
    
    return os.path.join(args.Dataset, "Results/Visualization",file_name + Puncuations.Dot + "VisLanes" + Puncuations.Dot+args.BackprojectionMethod +Puncuations.Dot + args.Detector+roi_str+ Puncuations.Dot + args.Tracker + Puncuations.Dot +SubTaskExt.VisTracking)
    
def add_opt_bw_path(args):
    args.OptBWImage = get_opt_bw_image(args)
    args.OptBWGround = get_opt_bw_ground(args)
    return args
def add_calculate_distance_path(args):
    args.CalculateDistanceVisPth=get_calculate_distance_vis_pth(args)
    return args
def add_calculate_speed_path(args):
    args.CalculateSpeedVisPth=get_calculate_speed_vis_pth(args)
    return args
def add_VisAll_path(args):
    args.VisAllPth=get_vis_all_path(args)
    return args
def add_visclass_path(args):
    args.VisClassPth=get_vis_class_path(args)
    return args
def add_vislanes_path(args):
    args.VisLanesPth=get_vis_lanes_path(args)
    return args
    
def get_GT_path(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    gt_path = os.path.join(args.Dataset, file_name+".gt.txt")
    return gt_path
def get_3DGT_path(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    gt_path = os.path.join(args.Dataset, file_name+".gt3d.json")
    return gt_path

def get_GT_json_path(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    gt_path = os.path.join(args.Dataset, file_name+".gt.json")
    return gt_path
    

def get_detections_mask_path(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    mask_path = os.path.join(args.Dataset, file_name+".detections.mask.npz")
    return mask_path
def get_gt_mask_path(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, file_name + Puncuations.Dot + "mask.npz")
def get_detection_mask_vis_path(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    mask_path = os.path.join(args.Dataset,"Results/Visualization", file_name+ ".VisDetectionMask.png")
    return mask_path
    
    
def add_GT_path_to_args(args):
    args.GT = get_GT_path(args)
    return args

def add_3DGT_path_to_args(args):
    args.GT3D = get_3DGT_path(args)
    # args.INTRINSICS_PATH        = get_intrinsics_path(args)
    # args.EXTRINSICS_PATH        = get_extrinsics_path(args)    
    return args

def add_GT_Json_path_to_args(args):
    args.GTJson=get_GT_json_path(args)
    return args
def get_extracted_image_dir(args):
     return os.path.join(args.Dataset, "Results/Images/")

def add_images_folder_to_args(args):
    args.ExtractedImageDirectory = get_extracted_image_dir(args)
    return args

def adjust_args_with_params(args):
    if args.CountMetric == "knn" or args.CountMetric == "gknn" :
        args.CountMetric += f".k={args.K}"

    if not args.DetectorVersion == "":
        args.Detector += "." + args.DetectorVersion

    if args.SAHI:
        args.Detector += f".SAHI-{args.SahiPatchSize}-\
            {str(args.SahiPatchOverlapRatio).replace('.', 'D')}-{str(args.SahiNMSTh).replace('.', 'D')}"

    return args

def revert_args_with_params(args):
    if args.CountMetric is not None:
        args.CountMetric = args.CountMetric.split(".")[0]

    if not args.DetectorVersion == "":
        args.Detector  = args.Detector.split(".")[0]

    if args.SAHI:
        args.Detector  = args.Detector.split(".")[0]

    return args

def get_check_point_dir(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/CheckPoints")

def get_detector_check_point_dir(args):
    file_name, file_ext = os.path.splitext(args.Video)
    file_name = file_name.split("/")[-1]
    return os.path.join(args.Dataset, "Results/CheckPoints", args.Detector)

def add_check_points_path_to_args(args):
    args.CheckPointDir = get_check_point_dir(args)
    if args.Detector is not None:
        args.DetectorCheckPointDir = get_detector_check_point_dir(args)
    return args

def flatten_args(args):
    """
    recursive func to flatten a nexted list of args
    """
    if not isinstance(args, list):
        return [args]
    args_flat = []
    for arg in args:
        arg_flat = flatten_args(arg)
        args_flat += arg_flat
    return args_flat

def get_sub_args(args):
    sp = args.Dataset
    sub_paths = []
    sub_ids = []
    for cl in os.listdir(sp):
        if (not cl.startswith(".")) and\
           (not cl == "Results") and\
           os.path.isdir(os.path.join(sp, cl)):
            
            sub_paths.append(os.path.join(sp, cl))
            sub_ids.append(cl)
            
    args_sub = []
    for sub, sub_id in zip(sub_paths, sub_ids):
        temp_args = copy.deepcopy(args)
        temp_args.Dataset = sub
        temp_args.SubID = sub_id
        args_sub.append(temp_args)
    return args_sub

def get_args(args):
    args.SubID = args.Dataset.split("/")[-1]
    args = complete_args(args)
    check_config(args)
    return args


def add_roi_paths_to_args(args):
    
    roi_dict={}
    for r in range(0,len(args.Rois),2):
        roi_dict[args.Rois[r]]=args.Rois[r+1]
        
    if(args.Dataset[-3:] in roi_dict):
        args.roi_num=roi_dict[args.Dataset[-3:]]
        args.RoiNpz= get_roi_npz_path_from_args(args)
        args.DetectionRoiDetectorPath = get_detection_roi_path_with_detector_from_args(args)
        args.DetectionDetectorPath = get_detection_roi_path_with_detector_from_args(args)
        args.DetectionPkl=get_detection_roi_pkl(args)
        args.DetectionPklBackUp=get_detection_roi_pkl_back_up(args)
        args.VisDetectionPth=get_vis_detection_roi_path_from_args(args)
        if args.Tracker is not None:
            args.TrackingPth = get_tracking_roi_path_from_args(args)
            args.TrackingPkl = get_tracking_roi_pkl_from_args(args)
            args.TrackingPklBackUp = get_tracking_roi_pkl_bu_from_args(args)
            args.VisTrackingPth = get_vis_tracking_roi_path_from_args(args)
        # args.DetectionRoiEvalPath= get_detect_eval_roi_save_path(args)
    return args
def add_detection_mask_path_to_args(args):
    args.DetectionMask=get_detections_mask_path(args)
    args.DetectionMaskVis=get_detection_mask_vis_path(args)
    return args
def get_args_gt(args):
    args_gt = copy.deepcopy(args)
    # args.GTDetector="GTHW7"
    args_gt.Detector = args.GTDetector
    
    args_gt.DetectorVersion = ""
    # args.GTTracker="GTHW7"
    args_gt.Tracker = args.GTTracker
    args_gt.Video=None
    args_gt.ContactPoint = args_gt.GTContactPoint
    args_gt=complete_args(args_gt)
    return get_args(args_gt)

def get_args_gt3D(args):
    args_gt3D = copy.deepcopy(args)
    args_gt3D.Detector = args.GTDetector3D
    args_gt3D.DetectorVersion = ""
    args_gt3D.Tracker = args.GTTracker3D
    args_gt3D.Video=None
    args_gt3D.ContactPoint = args_gt3D.GT3DContactPoint
    args_gt3D=complete_args(args_gt3D)
    return get_args(args_gt3D)

def get_args_mc(args):
    args_mc = get_sub_args(args)
    for i, arg_c in enumerate(args_mc):
        arg_c = get_args(arg_c)
        args_mc[i] = arg_c

    args.Video = args.Dataset
    args.ROIFromTop = False
    args = get_args(args)
    args = complete_args_mc(args)
    args.MetaData = args_mc[0].MetaData
    args.HomographyNPY = args_mc[0].HomographyNPY
    args.HomographyTopView = args_mc[0].HomographyTopView

    return args, args_mc

def get_args_ms(args):
    args_ms = get_sub_args(args)
    args_mcs = []
    for i, arg_s in enumerate(args_ms):
        arg_s, arg_s_mc = get_args_mc(arg_s)
        args_ms[i] = arg_s
        args_mcs.append(arg_s_mc)

    args.Video = args.Dataset
    args.ROIFromTop = False
    args = get_args(args)
    args = complete_args_ms(args)
    args.MetaData  = args_ms[0].MetaData
    args.HomographyNPY = args_ms[0].HomographyNPY
    args.HomographyTopView = args_ms[0].HomographyTopView

    return args, args_ms, args_mcs

def get_args_mp(args):
    args_mp = get_sub_args(args)
    args_mss = []
    args_mcs = []
    for i, arg_m in enumerate(args_mp):
        args_m, arg_m_mss, arg_m_mcs = get_args_ms(arg_m)
        args_mp[i] = args_m
        args_mss.append(arg_m_mss)
        args_mcs.append(arg_m_mcs)

    args.Video = args.Dataset
    args.ROIFromTop = False
    args = get_args(args)
    args = complete_args_mp(args)
    args.MetaData = args_mp[0].MetaData
    args.HomographyNPY = args_mp[0].HomographyNPY
    args.HomographyTopView = args_mp[0].HomographyTopView

    return args, args_mp, args_mss, args_mcs

def get_args_mp_gt(args):
    args = copy.deepcopy(args)
    # modify stuff before passing
    if args.GTDetector is not None:
        args.Detector = args.GTDetector
        args.DetectorVersion = ""
        
    return get_args_mp(args)

def complete_args(args):
    args = adjust_args_with_params(args)

    if args.TrackLabelingGUI or args.ExtractCommonTracks:
        args.Meter=True

    if args.Video is None:
        # if Video path was not specified by the user grab a video from dataset
        args = add_videos_to_args(args)

    if (not args.Detector is None) or args.DetPostProc or args.ConvertDetsToCOCO or args.FineTune:
        args = add_detection_pathes_to_args(args)
        args = add_vis_detection_path_to_args(args)
        
    if not args.Tracker is None:
        args = add_tracking_path_to_args(args)
        args = add_vis_tracking_path_to_args(args)
        args = add_vis_top_tracking_path_to_args(args)
        args = add_tracking_pkl_to_args(args)
    if args.Rois is not None:
        args=add_roi_paths_to_args(args)
        
    if args.TrackEval:
        args = add_track_eval_save_path(args)
    if args.DetectEval:
        args= add_detect_eval_save_path(args)
    try:
        args = add_metadata_to_args(args)
    except:
        pass

    try:
        args = add_calibration_related_path_to_args(args)
    except:
        pass

    args = add_GT_path_to_args(args)
    args = add_3DGT_path_to_args(args)
    args = add_GT_Json_path_to_args(args)
    args = add_images_folder_to_args(args)

    args = add_check_points_path_to_args(args)

    if args.HomographyGUI or args.Homography or args.VisHomographyGUI or\
        args.VisTrajectories or args.VisLabelledTrajectories or args.Cluster or\
        args.TrackPostProc or args.Count or args.VisROI or args.Meter or\
        args.VisTrackTop or args.FindOptimalKDEBW or args.VisCPTop or\
        args.EvalCount    or args.IntegrateCountsMC or\
        args.EvalCountMSfromMC:
            # or args.DetPostProc or args.TrackEval
            
        args = add_homographygui_related_path_to_args(args)

    if args.Homography or args.VisTrajectories or args.VisLabelledTrajectories or\
        args.Meter or args.Cluster or args.TrackPostProc or args.Count or args.Meter or\
        args.VisTrackTop or args.FindOptimalKDEBW or args.VisContactPoint or args.VisCPTop or\
        args.EvalCount  or args.EvalContactPoitnSelection or\
        args.IntegrateCountsMC or args.EvalCountMSfromMC:
        # or args.DetPostProc or args.TrackEval\
            
        args = add_homography_related_path_to_args(args)
        args = add_dsm_related_path_to_args(args)

    if args.VisHomographyGUI or args.VisLabelledTrajectories or args.Meter or args.FindOptimalKDEBW:
        args = add_vishomography_path_to_args(args)

    if args.TrackLabelingGUI or args.VisLabelledTrajectories or args.Meter or\
        args.ExtractCommonTracks or args.Count or args.EvalCount:
        args = add_tracklabelling_export_to_args(args)

    if args.VisTrajectories:
        args = add_plot_all_traj_pth_to_args(args)

    if args.VisLabelledTrajectories:
        args = add_vis_labelled_tracks_pth_to_args(args)

    if args.Meter or args.Count or args.Cluster or args.TrackPostProc or args.FindOptimalKDEBW or\
        args.EvalCount:
        args = add_meter_path_to_args(args)

    if args.Count or args.VisTrackMoI or args.AverageCountsMC or\
         args.EvalCount or args.IntegrateCountsMC or args.EvalCountMSfromMC or args.EvalCountMC:
        args = add_count_path_to_args(args)

    if args.Cluster or args.TrackLabelingGUI:
        args = add_clustering_related_pth_to_args(args)

    if args.VisROI:
        args = add_visroi_path_to_args(args)

    if args.VisTrackMoI:
        args = add_vis_tracking_moi_path_to_args(args)

    if args.Count or args.EvalCount:
        args = add_cached_counter_path_to_args(args)
        args = add_density_path_to_args(args)
        
    if args.Segment or args.SegPostProc or args.Homography:
        args = add_segmentation_path_to_args(args)
        args = add_vis_segment_path_to_args(args)
    if args.MaskDetections:
        args=add_detection_mask_path_to_args(args)
    if args.FindOptimalKDEBW:
        args = add_opt_bw_path(args)
    if args.MaskGT:
        args= add_gt_mask_path_to_args(args)        
    if args.CalcDistance:
        args= add_calculate_distance_path(args)
    if args.CalcSpeed:
        args= add_calculate_speed_path(args)
    if args.VisAll:
        args= add_VisAll_path(args)
    if args.VisClass:
        args=add_visclass_path(args)
    if args.FindLanes:
        args=add_vislanes_path(args)
    args = add_correct_roi_to_args(args)
    args = revert_args_with_params(args)
    return args

def complete_args_mc(args):
    # if args.AverageCountsMC or args.EvalCountMC:
    #     args = add_count_path_to_args(args)
    return args

def complete_args_ms(args):
    return args

def complete_args_mp(args):
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
    segment_path    = os.path.join(results_path, "Segment")
    dsm_path    = os.path.join(results_path, "DSM")
    images_path = os.path.join(results_path, "Images")
    check_points_path = os.path.join(results_path, "CheckPoints")

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
    try: os.system(f"mkdir -p {segment_path}")
    except: pass
    try: os.system(f"mkdir -p {dsm_path}")
    except: pass
    try: os.system(f"mkdir -p {images_path}")
    except: pass
    try: os.system(f"mkdir -p {check_points_path}")
    except: pass

def get_conda_envs():
    stream = os.popen("conda env list")
    output = stream.read()
    a=output.split()
    a.remove("*")
    a.remove("#")
    a.remove("#")
    a.remove("conda")
    a.remove("environments:")
    return a[::2]

def get_numgpus_torch(env_name):
    stream = os.popen(f"conda run -n {env_name} --live-stream python -c 'import torch; ngpus=torch.cuda.device_count(); print(ngpus);'")
    output = stream.read()
    ngpus = int(output)
    return ngpus

def get_available_port(host='localhost'):
    with socket.socket() as sock:
        sock.bind((host, 0))
        port = sock.getsockname()[1]
    return port
    
def make_conda_env(env_name, libs=""):
    os.system(f"conda create -n {env_name} -y "+libs)

def activate_conda_env(env_name):
    os.system(f"conda activate {env_name}")

def deactivate_conda_env(env_name):
    os.system(f"conda deactivate")

def conda_pyrun(env_name, exec_file, args):
    os.system(f"conda run -n {env_name} --live-stream python3 \"{exec_file}\" '{json.dumps(dict(vars(args)))}'")

def download_url_to(url, path):
    # make sure that path is valid
    r = requests.get(url, allow_redirects=True)
    open(path, 'wb').write(r.content)

def save_frame_from_video(args):
    video_path = args.Video
    image_out_path = args.HomographyStreetView
    chosen_frame = args.Frame # leave it this way for now
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

def direction_vector(traj_a):
    x = traj_a[-1] - traj_a[0]
    if np.linalg.norm(x) > 0:
        return x/np.linalg.norm(x)
    else:
        return x

def cosine_distance(v1, v2):
    if np.linalg.norm(v1) > 0:
        v1 = v1/np.linalg.norm(v1)
    if np.linalg.norm(v2)>0:
        v2 = v2 / np.linalg.norm(v2)
    return  1 - np.dot(v1, v2)

# Similarity Metrics (used in counting and clustering

def cmm_dist(traj_a, traj_b):
    libfile = "./counting/cmm_truncate_linux/build/lib.linux-x86_64-3.7/cmm.cpython-37m-x86_64-linux-gnu.so"
    cmmlib = ctypes.CDLL(libfile)
    # 2. tell Python the argument and result types of function cmm
    cmmlib.cmm_truncate_sides.restype = ctypes.c_double
    cmmlib.cmm_truncate_sides.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64),
                                                np.ctypeslib.ndpointer(dtype=np.float64),
                                                np.ctypeslib.ndpointer(dtype=np.float64),
                                                np.ctypeslib.ndpointer(dtype=np.float64),
                                                ctypes.c_int, ctypes.c_int]

    traj_a = traj_a.astype(dtype=np.float64)
    traj_b = traj_b.astype(dtype=np.float64)
    if traj_a.shape[0] >= traj_b.shape[0]:
        c = cmmlib.cmm_truncate_sides(traj_a[:, 0], traj_a[:, 1], traj_b[:, 0], traj_b[:, 1], traj_a.shape[0],
                                        traj_b.shape[0])
    else:
        c = cmmlib.cmm_truncate_sides(traj_b[:, 0], traj_b[:, 1], traj_a[:, 0], traj_a[:, 1], traj_b.shape[0],
                                        traj_a.shape[0])

    c = np.abs(c)
    return c

def dc_dist(traj_a, traj_b):
    d_a = direction_vector(traj_a)
    d_b = direction_vector(traj_b)
    return cosine_distance(d_a, d_b)

def ccmm_dist(traj_a, traj_b):
    c1 = CMM.cmm_dist(traj_a, traj_b)
    c2 = dc_dist(traj_a, traj_b)
    return c1*c2

def tccmm_dist(traj_a, traj_b):
    c1 = CMM.cmm_dist(traj_a, traj_b)
    cmix =tc_dist(traj_a, traj_b)
    return c1*cmix

def tc_dist(traj_a, traj_b):
    c2 = dc_dist(traj_a, traj_b)
    traj_a_mid = traj_a[int(len(traj_a)/2)]
    traj_b_mid = traj_b[int(len(traj_b)/2)]
    traj_a_1 = traj_a_mid - traj_a[0]
    traj_a_2 = traj_a[-1] - traj_a_mid
    traj_b_1 = traj_b_mid - traj_b[0]
    traj_b_2 = traj_b[-1] - traj_b_mid
    c3 = cosine_distance(traj_a_1, traj_b_1)
    c4 = cosine_distance(traj_a_2, traj_b_2)
    return c2+c3+c4

def minvc_dist(traj_a, traj_b):
    return np.linalg.norm(traj_a[0] - traj_b[0]) + np.linalg.norm(traj_a[-1] - traj_b[-1])

def ptcos_dist(traj_a, traj_b):
    return tc_dist(traj_a, traj_b) * minvc_dist(traj_a, traj_b)

def hausdorff_directed(traj_a, traj_b, dist_func = lambda va, vb: np.linalg.norm(va-vb)):
    diffs = traj_a[:, np.newaxis, :] - traj_b[np.newaxis, :, :]
    # Compute the squared distances
    squared_diffs = diffs ** 2
    # Sum the squared differences across the columns to get squared distances
    squared_distances = np.sum(squared_diffs, axis=2)
    # Take the square root to get the Euclidean distances
    D = np.sqrt(squared_distances)
    mins = np.min(D, axis = 1)
    max_dist = np.max(mins)
    return max_dist
            
def hausdorff_dist(traj_a, traj_b):
    d_ab = hausdorff_directed(traj_a, traj_b)
    d_ba = hausdorff_directed(traj_b, traj_a)
    return max(d_ab, d_ba)

Metric_Dict = {
    "cmm":       cmm_dist,
    "ccmm":      ccmm_dist,
    "tccmm":     tccmm_dist,
    "cos":       dc_dist,
    "tcos":      tc_dist,
    "hausdorff": hausdorff_dist,
    "ptcos"    : ptcos_dist,
    "gcmm":       cmm_dist,
    "gccmm":      ccmm_dist,
    "gtccmm":     tccmm_dist,
    "gcos":       dc_dist,
    "gtcos":      tc_dist,
    "ghausdorff": hausdorff_dist,
    "gptcos"    : ptcos_dist
    }

# A Python3 program to find if 2 given line segments intersect or not

class Point:
	def __init__(self, p):
		self.x = p[0]
		self.y = p[1]

# Given three collinear points p, q, r, the function checks if
# point q lies on line segment 'pr'
def onSegment(p, q, r):
	if ( (q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and
		(q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))):
		return True
	return False

def orientation(p, q, r):
	# to find the orientation of an ordered triplet (p,q,r)
	# function returns the following values:
	# 0 : Collinear points
	# 1 : Clockwise points
	# 2 : Counterclockwise
	
	# See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/
	# for details of below formula.
	
	val = ((q.y - p.y) * (r.x - q.x)) - ((q.x - p.x) * (r.y - q.y))
	if (val > 0):
		
		# Clockwise orientation
		return 1
	elif (val < 0):
		
		# Counterclockwise orientation
		return 2
	else:
		
		# Collinear orientation
		return 0

# The main function that returns true if
# the line segment 'p1q1' and 'p2q2' intersect.
def doIntersect(p1,q1,p2,q2):
	
	# Find the 4 orientations required for
	# the general and special cases
	o1 = orientation(p1, q1, p2)
	o2 = orientation(p1, q1, q2)
	o3 = orientation(p2, q2, p1)
	o4 = orientation(p2, q2, q1)

	# General case
	if ((o1 != o2) and (o3 != o4)):
		return True

	# Special Cases

	# p1 , q1 and p2 are collinear and p2 lies on segment p1q1
	if ((o1 == 0) and onSegment(p1, p2, q1)):
		return True

	# p1 , q1 and q2 are collinear and q2 lies on segment p1q1
	if ((o2 == 0) and onSegment(p1, q2, q1)):
		return True

	# p2 , q2 and p1 are collinear and p1 lies on segment p2q2
	if ((o3 == 0) and onSegment(p2, p1, q2)):
		return True

	# p2 , q2 and q1 are collinear and q1 lies on segment p2q2
	if ((o4 == 0) and onSegment(p2, q1, q2)):
		return True

	# If none of the cases
	return False