# Author: Sajjad P. Savaoji April 27 2022
# This py file will run the whole transplan piple line
# The pipe line consists of the following components
# 1. Preprocessing(MATLAB) 2. Detection  3. Tracking  4. Homograhy  5. Counting/Clustering
# Detectors are placed in the "Detectors" sub-folder
# Trackers are placed  in the "Trackers" sub-folder
# This piple line comes with two GUI features: 1. Homography GUI(to find homography matrix)  2. Track-Labeling GUI(annotation)

# import libs
from Libs import *
from Utils import *
from Detect import detect, visdetect,detectpostproc, visroi
from Track import track, vistrack, trackpostproc, vistrackmoi, vistracktop
from Homography import homographygui
from Homography import reproject
from Homography import vishomographygui
from Homography import vis_reprojected_tracks
from TrackLabeling import tracklabelinggui, vis_labelled_tracks, extract_common_tracks
from Evaluate import evaluate
from Maps import pix2meter
from counting import counting
from counting.counting import find_opt_bw, eval_count
from Clustering import cluster
from CountingMC import AverageCountsMC
from Segment import segment, vis_segment

def Preprocess(args):
    if args.Preprocess:
        return FailLog("Preprocessing part should be done in MARLAB for now")
    else: return WarningLog("skipped preprocess subtask")

def Detect(args):
    if args.Detect:
        print(ProcLog("Detection in Process"))
        log = detect(args)
        return log
    else: return WarningLog("skipped detection subtask")

def DetPostProc(args):
    if args.DetPostProc:
        print(ProcLog("Detection post processing in execution"))
        log = detectpostproc(args)
        return log
    else: return WarningLog("skipped Detection post processing subtask")

def VisROI(args):
    if args.VisROI:
        print(ProcLog("Visualizing the ROI is in process"))
        log = visroi(args)
        return log 
    else: return WarningLog("skipped VisROI Part")

def VisDetect(args):
    if args.VisDetect:
        print(ProcLog("Viz-Detection in Process"))
        log = visdetect(args)
        return log
    else: return WarningLog("skipped viz-detection subtask")

def Track(args):
    if args.Track:
        print(ProcLog("Tracking in Process"))
        log = track(args)
        return log
    else: return WarningLog("skipped tracking subtask")

def VisTrack(args):
    if args.VisTrack:
        print(ProcLog("Vis-Tracking in Process"))
        log = vistrack(args)
        return log
    else: return WarningLog("skipped vis-tracking subtask")

def HomographyGUI(args):
    if args.HomographyGUI:
        print(ProcLog("Homography GUI in Process"))
        log = homographygui(args)
        return log
    else: return WarningLog("skipped homography GUI subtask")

def VisHomographyGUI(args):
    if args.VisHomographyGUI:
        print(ProcLog("VisHomography GUI in Process"))
        log = vishomographygui(args)
        return log
    else: return WarningLog("skipped vis homography GUI subtask")

def Homography(args, from_back_up = False):
    if args.Homography:
        print(ProcLog("Reprojection with Homographies in Process"))
        log = reproject(args, method=args.BackprojectionMethod, source = args.BackprojectSource, from_back_up=from_back_up)
        return log
    else: return WarningLog("skipped homography subtask")

def FindOptBW(args):
    if args.FindOptimalKDEBW:
        print(ProcLog("finding optimal KDE BW"))
        log = find_opt_bw(args)
        return log
    else: return WarningLog("skipped find opt bw")

def TrackLabelingGUI(args):
    if args.TrackLabelingGUI:
        print(ProcLog("Track Labeling GUI in Process"))
        log = tracklabelinggui(args)
        log_temp = pix2meter(args)
        print(log_temp)
        return log
    else: return WarningLog("skipped track labelling subtask")

def VisTrajectories(args):
    if args.VisTrajectories:
        print(ProcLog("VisTrajectories in Process"))
        log = vis_reprojected_tracks(args)
        return log
    else: return WarningLog("skipped plotting all tracks")

def VisLabelledTrajectories(args):
    if args.VisLabelledTrajectories:
        print(ProcLog("Vis Labeled trajectories in Process"))
        log = vis_labelled_tracks(args)
        return log
    else: return WarningLog("skipped plotting labelled tracks")

def Pix2Meter(args):
    if args.Meter:
        print(ProcLog("Converting to meter in Process"))
        log = pix2meter(args)
        return log
    else: return WarningLog("skipped changing pixel values to meter values")

def TrackPostProc(args):
    if args.TrackPostProc:
        args.Meter , args.Homography = True , True
        args = complete_args(args)
        print(ProcLog("Track Post Processing in execution"))
        log = trackpostproc(args)
        return log
    else: return WarningLog("skipped track post processing")

def Count(args):
    if args.Count:
        print(ProcLog("Counting in Process"))
        log = counting.main(args)
        return log
    else: return WarningLog("skipped counting subtask")

def Cluster(args):
    if args.Cluster:
        print(ProcLog("Clustering in Process"))
        log = cluster(args)
        return log
    else: return WarningLog("skipped clustering subtask")

def VisTrackMoI(args):
    if args.VisTrackMoI:
        print(ProcLog("Vis Tracking with MoI labels in Process"))
        log = vistrackmoi(args)
        return log
    else: return WarningLog("skipped clustering subtask")

def ExtractCommonTracks(args):
    if args.ExtractCommonTracks:
        print(ProcLog("Extract Common Trajectories from video"))
        log = extract_common_tracks(args)
        log_temp = pix2meter(args)
        print(log_temp)
        return log
    else: return WarningLog("skipped extract common track subtask")

def VisTrackTop(args):
    if args.VisTrackTop:
        print(ProcLog("Vis Top Tracking"))
        log = vistracktop(args)
        return log
    else: return WarningLog("skipped vis tracking from top")
def Evaluate(args):
    if args.Eval:
        print(ProcLog("Evaluate Tracking"))
        log = evaluate(args)
        return log
    else: return WarningLog("skipped vis tracking from top")


def AverageCounts(args, args_mc):
    if args.AverageCountsMC:
        print(ProcLog("Averaging Counting on all cameras"))
        log = AverageCountsMC(args, args_mc)
        return log
    else:
        return WarningLog("skipped averaging counts")

def EvalCountMC(args, args_mc):
    if args.EvalCountMC:
        print(ProcLog("Evaluating counts MC"))
        args.MetaData = args_mc[0].MetaData
        log = eval_count(args)
        return log
    else:
        return WarningLog("skipped evaluating counts")
    
def Segment(args):
    if args.Segment:
        print(ProcLog("segment images"))
        log = segment(args)
        return log
    else: return WarningLog("skipped segmentation subtask")

def VisSegment(args):
    if args.VisSegment:
        print(ProcLog("visulaize segmentation masks"))
        log = vis_segment(args)
        return log
    else: return WarningLog("skipped vis segmentations")

def main(args):
    # Pass the args to each subtask
    # Each subtask will validate its own inputs
    subtasks = [Preprocess,
                HomographyGUI, VisHomographyGUI, VisROI,
                Segment, VisSegment,
                Detect, DetPostProc, VisDetect,
                Track, VisTrack, Homography, Pix2Meter, TrackPostProc, VisTrajectories, VisTrackTop,
                FindOptBW, Cluster, ExtractCommonTracks, TrackLabelingGUI, VisLabelledTrajectories,
                Count, VisTrackMoI, Evaluate]
    for subtask in subtasks:
        log = subtask(args)
        if not isinstance(log, WarningLog):
            print(log)

def main_mc(args, args_mc):
    # Pass the args to each subtask
    # Each subtask will validate its own inputs

    subtasks = [AverageCounts, EvalCountMC]
    for subtask in subtasks:
        log = subtask(args,args_mc)
        if not isinstance(log, WarningLog):
            print(log)
    
if __name__ == "__main__":
    # ferch the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--Dataset", help="Path to a rep containing video files", type=str)
    parser.add_argument("--Video", help="a list of video pathes in one repo", type=list)
    parser.add_argument("--Preprocess", help="If preprocess inputs first", action="store_true")
    parser.add_argument("--Detector", help="Name of detector to be used", type=str)
    parser.add_argument("--Tracker", help="Name of tracker to be used", type=str)
    parser.add_argument("--Detect", help="If perform detection", action="store_true")
    parser.add_argument("--VisDetect", help="If create video of detections", action="store_true")
    parser.add_argument("--Track", help="If perform tracking", action="store_true")
    parser.add_argument("--VisTrack", help="If create video visualization of tracking", action="store_true")
    parser.add_argument("--HomographyGUI", help="If pop-up homography GUI", action="store_true")
    parser.add_argument("--Frame", help="visualize the N first frame instead of all of them", type=int, default=10)
    parser.add_argument("--VisHomographyGUI", help="to visualize homography-gui results", action="store_true")
    parser.add_argument("--TrackLabelingGUI", help="If pop-up Track Labeling GUI", action="store_true")
    parser.add_argument("--Homography", help="If perform backkprojection using homography matrix", action="store_true")
    parser.add_argument("--VisTrajectories", help="If plot all the tracks", action="store_true")
    parser.add_argument("--VisLabelledTrajectories", help="If plot labelled tracks", action="store_true")
    parser.add_argument("--Count", help="If count the objects for each MOI", action="store_true")
    parser.add_argument("--CountVisPrompt", help="visualize each query track after classification", action="store_true")
    parser.add_argument("--CountVisDensity", help="if visualize densities of trained KDE", action="store_true")
    parser.add_argument("--CountMetric", help="name of the metric used in counting part",type=str)
    parser.add_argument("--EvalCount", help="Evaluate the Counting Result you got from Counting part", action="store_true")
    parser.add_argument("--Meter", help="convert reprojected track coordinated into meter", action="store_true")
    parser.add_argument("--Cluster", help="if to perform clustering", action="store_true")
    parser.add_argument("--ClusterMetric", help="The name of the distance metric used to compute the similarity metrics",type=str)
    parser.add_argument("--ClusteringAlgo", help="name of the clustering algorithm to be performed",type=str)
    parser.add_argument("--DetPostProc", help="if to perform detection post processings", action="store_true")
    parser.add_argument("--DetTh", help="the threshold for detection post processing", type=float)
    parser.add_argument('--classes_to_keep', nargs='+', type=float, default=[])
    parser.add_argument("--DetMask", help="if to remove bboxes out of ROI", action="store_true")
    parser.add_argument("--TrackPostProc", help="if to perform tracking post processings", action="store_true")
    parser.add_argument("--Interpolate", help="if to perform interpolation on tracks", action="store_true")
    parser.add_argument("--TrackTh", help="the threshold for short track removal in meter", type=float)
    parser.add_argument("--MaskROI", help="if to remove bboxes out of ROI", action="store_true")
    parser.add_argument("--RemoveInvalidTracks", help="remove tracks with less than 3 points", action="store_true")
    parser.add_argument("--SelectEndingInROI", help="select only those tracks that end in ROI", action="store_true")
    parser.add_argument("--SelectBeginInROI", help="select only those tracks that begin in ROI", action="store_true")
    parser.add_argument("--SelectDifEdgeInROI", help="remove tracks that begin and end in the same ROI region", action="store_true")
    parser.add_argument("--HasPointsInROI", help="select the tracks that have at least on point in the ROI", action="store_true")
    parser.add_argument("--CrossROI", help="select tracks that cross the edges of roi at least once", action="store_true")
    parser.add_argument("--CrossROIMulti", help="select tracks that cross multiple edges of the roi", action="store_true")
    parser.add_argument("--JustEnterROI", help="select tracks that cross multiple edges of the roi", action="store_true")
    parser.add_argument("--JustExitROI", help="select tracks that cross multiple edges of the roi", action="store_true")
    parser.add_argument("--WithinROI", help="select tracks that cross multiple edges of the roi", action="store_true")
    parser.add_argument("--ExitOrCrossROI", help="select tracks that either exit or cross multi roi", action="store_true")
    parser.add_argument("--MaskGPFrame", help="remove dets on tracks that are outside gp frame", action="store_true")
    parser.add_argument("--Eval", help="Evaluate the tracking single camera", action="store_true")
    parser.add_argument("--VisROI", help="visualize the selected ROI", action='store_true')
    parser.add_argument("--VisTrackMoI", help="visualize tracking with moi labels", action='store_true')
    parser.add_argument("--LabelledTrajectories", help=" a pkl file containint the labelled trajectories on the ground plane",type=str)
    parser.add_argument("--ExtractCommonTracks", help="instead of track labelling GUI extract common tracks automatically", action='store_true')
    parser.add_argument("--UseCachedCounter", help="use a pre-initialized cached counter object", action='store_true')
    parser.add_argument("--CacheCounter", help="Cache the counter after initialization", action='store_true')
    parser.add_argument("--VisTrackTop", help="Visualize tracks from top view", action='store_true')
    parser.add_argument("--CachedCounterPth", help="path to pre-initialized cached counter object", type=str)

    parser.add_argument("--ForNFrames", help="visualize the N first frame instead of all of them", type=int)
    parser.add_argument("--ResampleTH", help="the threshold to resample tracks with", type=float, default=2)

    parser.add_argument("--FindOptimalKDEBW", help="find the optimal KDE band width", action='store_true')

    parser.add_argument("--K", help="K in KNN classifier", type=int, default=1)

    parser.add_argument("--MultiCam", help="operating in multi-camera", action='store_true')
    parser.add_argument("--AverageCountsMC", help="averaging counting on MC", action='store_true')
    parser.add_argument("--EvalCountMC", help="Evaluate Counts MC", action="store_true")

    parser.add_argument("--TopView", help="seeting which topview to use. Options are [GoogleMap, OrthoPhoto]", type=str)
    parser.add_argument("--BackprojectSource", help="selecting which source to backproject form Options are [tracks, detections]", type=str)
    parser.add_argument("--BackprojectionMethod", help="Select back projection method  options = [Homography/DSM]", type=str)

    parser.add_argument("--Segment", help="perform segmentation and store results", action='store_true')
    parser.add_argument("--VisSegment", help="Vis Segmentation Masks", action='store_true')
    parser.add_argument("--Segmenter", help="model for segmentation", type=str, default="Null")


    args = parser.parse_args()

    # check if the opeerations should be performed cross camera
    if args.MultiCam:
        # duplicate args and set new DataSet path for each
        args_mc = modify_args_mc(args)
        args , args_mc = complete_args_mc(args, args_mc)
        main_mc(args, args_mc)
    
    else:
        args = complete_args(args)
        check_config(args)
        main(args)

