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
from Detect import detect, visdetect,detectpostproc, visroi, extract_images, detections_to_coco, fine_tune_detector_mp
from Track import track, vistrack, trackpostproc, vistrackmoi, vistracktop, calculate_distance
from Homography import homographygui
from Homography import reproject
from Homography import vishomographygui
from Homography import vis_reprojected_tracks, vis_contact_point, vis_contact_point_top, eval_contact_points
from TrackLabeling import tracklabelinggui, vis_labelled_tracks, extract_common_tracks, extract_common_tracks_multi
from Evaluate import evaluate_tracking, cvpr
from Maps import pix2meter
from counting import counting
from counting.counting import find_opt_bw, eval_count, eval_count_multi
from Clustering import cluster
from CountingMC import AverageCountsMC
from Segment import segment, vis_segment, SegmentPostProc

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

def VisLabelledTrajectoriesMulti(args, args_ms, args_mcs):
    if args.VisLabelledTrajectories:
        print(ProcLog("Vis Labeled trajectories in Process for MCMS"))
        for args_temp in flatten_args(args_mcs):
            vis_labelled_tracks(args_temp)
        vis_labelled_tracks(args)
        return SucLog("visualized all the labelled trajectories for all segments")
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

def CountMS(args, args_ms, args_mcs):
    if args.Count:
        print(ProcLog("Counting Multi in Process"))
        log = counting.mainMulti(args, args_mcs)
        return log
    else: return WarningLog("skipped counting subtask")

def EvalCount(args):
    if args.EvalCount:
        print(ProcLog("evaluating counting"))
        log = eval_count(args)
        return log
    else: return WarningLog("skipped eval counting subtask")

def EvalCountMS(args, args_ms, args_mcs):
    if args.EvalCount:
        print(ProcLog("evaluating counting MS"))
        log = eval_count_multi(args, args_mcs)
        return log
    else: return WarningLog("skipped eval counting subtask")
    
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
        log = extract_common_tracks(args, args.GP)
        log_temp = pix2meter(args)
        print(log_temp)
        return log
    else: return WarningLog("skipped extract common track subtask")

def ExtractCommonTracksMulti(args, args_ms, args_mcs):
    if args.ExtractCommonTracks:
        print(ProcLog("Extract Common Trajectories from multi segments and multi cameras"))
        log = extract_common_tracks_multi(args, args_mcs)
        for args_temp in flatten_args(args_mcs):
            log_temp = pix2meter(args_temp)
        log_temp_top = pix2meter(args)
        return log
    else: return WarningLog("skipped extract common track MCMS subtask")

def VisTrackTop(args):
    if args.VisTrackTop:
        print(ProcLog("Vis Top Tracking"))
        log = vistracktop(args)
        return log
    else: return WarningLog("skipped vis tracking from top")

def TrackEvaluate(args):
    if args.TrackEval:
        print(ProcLog("Evaluate Tracking"))
        log = evaluate_tracking(args, args)
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
    
def TrackEvaluateMC(args, args_mc):
    if args.TrackEval:
        print(ProcLog("Single Source Evaluate Tracking"))
        log = evaluate_tracking(args, args_mc)
        return log
    else: return WarningLog("skipped TrackEvaluateMC")
    
def Segment(args):
    if args.Segment:
        print(ProcLog("segment images"))
        log = segment(args)
        return log
    else: return WarningLog("skipped segmentation subtask")

def SegPostProc(args):
    if args.SegPostProc:
        print(ProcLog("segment post processing"))
        log = SegmentPostProc(args)
        return log
    else: return WarningLog("skipped segPostProc task")
# def CalculateDistances(args):
#     if args.CalcDistance:
#         print(ProcLog("calculing distance between vehicles and intersection"))
#         log= calculate_distance(args)
#         return log
#     else: return  WarningLog("skipped calc distance subtask")
def VisSegment(args):
    if args.VisSegment:
        print(ProcLog("visulaize segmentation masks"))
        log = vis_segment(args)
        return log
    else: return WarningLog("skipped vis segmentations")

def VisContactPoint(args):
    if args.VisContactPoint:
        print(ProcLog("visualizing the contact point"))
        log = vis_contact_point(args)
        return log
    else: return WarningLog("skipped vis contact point")

def EvalCPSelection(args):
    if args.EvalContactPoitnSelection:
        print(ProcLog("evaluating contact points"))
        log = eval_contact_points(args)
        return log
    else: return WarningLog("skipped contact point evaluation") 

def VisCPTop(args):
    if args.VisCPTop:
        print(ProcLog("vis cp top"))
        log = vis_contact_point_top(args)
        return log
    else: return WarningLog("skipped vis cp top")

def ExtractImages(args):
    if args.ExtractImages:
        print(ProcLog("extracting images"))
        log = extract_images(args)
        return log
    else: return WarningLog("skipped extarcting imagages")

def ConvertDetsToCOCO(args):
    if args.ConvertDetsToCOCO:
        print(ProcLog("converting detections to coco format"))
        log  = detections_to_coco(args, args)
        return log
    else: return WarningLog("skipped converting to coco format")

def ConvertDetsToCOCO_MS(args, args_ms, args_mcs):
    if args.ConvertDetsToCOCO:
        print(ProcLog(f"converting detections to coco format for the {args.Dataset} split"))
        log  = detections_to_coco(args, args_mcs)
        return log
    else: return WarningLog("skipped converting to coco format")

def TrackEvaluateMS(args, args_ms, args_mcs):
    if args.TrackEval:
        print(ProcLog("Single Source Evaluate Tracking"))
        log = evaluate_tracking(args, args_mcs)
        return log
    else: return WarningLog("skipped TrackEvaluate on MS")

def FineTuneDetectorMP(args, args_mp, args_mss, args_mcs):
    if args.FineTune:
        print(ProcLog(f"Finetunning detectors"))
        log = fine_tune_detector_mp(args, args_mp, args_mss, args_mcs)
        return log
    else: return WarningLog("skipped fine tunning detectors")

def TrackEvaluateMP(args, args_mp, args_mss, args_mcs):
    if args.TrackEval:
        print(ProcLog("Single Source Evaluate Tracking"))
        log = evaluate_tracking(args, args_mcs)
        return log
    else: return WarningLog("skipped TrackEvaluate on MP")

def CVPRMP(args, args_mp, args_mss, args_mcs):
    if args.CVPR:
        print(ProcLog("cvpr log"))
        print(len(args_mcs))
        log = cvpr(args, args_mcs)
        return log
    else: return WarningLog("skipped CVPR on MP")

def main(args):
    # main for one video
    subtasks = [Preprocess, ExtractImages,
                HomographyGUI, VisHomographyGUI, VisROI,
                Segment, SegPostProc, VisSegment,
                Detect, DetPostProc, VisDetect, ConvertDetsToCOCO,
                Track, Homography, Pix2Meter, TrackPostProc, TrackEvaluate,
                VisTrack, VisTrajectories, VisTrackTop,
                VisContactPoint, VisCPTop, EvalCPSelection,
                FindOptBW, Cluster, ExtractCommonTracks, TrackLabelingGUI, VisLabelledTrajectories,
                Count, EvalCount, VisTrackMoI]
    for subtask in subtasks:
        log = subtask(args)
        if not isinstance(log, WarningLog):
            print(log)

def main_mc(args, args_mc):
    # main for multi camera

    subtasks = [TrackEvaluateMC, AverageCounts, EvalCountMC]
    for subtask in subtasks:
        log = subtask(args,args_mc)
        if not isinstance(log, WarningLog):
            print(log)

def main_ms(args, args_ms, args_mcs):
    # main for multi segments
    subtasks = [ConvertDetsToCOCO_MS, TrackEvaluateMS,
                ExtractCommonTracksMulti, VisLabelledTrajectoriesMulti,
                CountMS, EvalCountMS]

    for sub in subtasks:
        log = sub(args, args_ms, args_mcs)
        if not isinstance(log, WarningLog):
            print(log)

def main_mp(args, args_mp, args_mss, args_mcs):
    # main for multi parts
    subtasks = [FineTuneDetectorMP, TrackEvaluateMP, CVPRMP]
    for sub in subtasks:
        log = sub(args, args_mp, args_mss, args_mcs)
        if not isinstance(log, WarningLog):
            print(log)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--Dataset", help="Path to a rep containing video files", type=str)
    parser.add_argument("--Video", help="a list of video pathes in one repo", type=list)
    parser.add_argument("--Preprocess", help="If preprocess inputs first", action="store_true")
    parser.add_argument("--Detector", help="Name of detector to be used", type=str)
    parser.add_argument("--Tracker", help="Name of tracker to be used", type=str)
    parser.add_argument("--Detect", help="If perform detection", action="store_true")
    parser.add_argument("--DetectorVersion", help="select detector version to load. Each version corresponds to a checkpoint file.", type=str, default="")
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
    parser.add_argument("--InterpolateTh", help="the threshold for interpolation", type=float)
    parser.add_argument("--TrackTh", help="the threshold for short track removal in meter", type=float)
    parser.add_argument("--MaskROI", help="if to remove bboxes out of ROI", action="store_true")
    parser.add_argument("--RemoveInvalidTracks", help="remove tracks with less than 3 points", action="store_true")
    parser.add_argument("--SelectEndingInROI", help="select only those tracks that end in ROI", action="store_true")
    parser.add_argument("--SelectBeginInROI", help="select only those tracks that begin in ROI", action="store_true")
    parser.add_argument("--SelectDifEdgeInROI", help="remove tracks that begin and end in the same ROI region", action="store_true")
    parser.add_argument("--HasPointsInROI", help="select the tracks that have at least on point in the ROI", action="store_true")
    parser.add_argument("--MovesInROI", help="if the track has at least 2 points after sampling in ROI", action="store_true")
    parser.add_argument("--CrossROI", help="select tracks that cross the edges of roi at least once", action="store_true")
    parser.add_argument("--CrossROIMulti", help="select tracks that cross multiple edges of the roi", action="store_true")
    parser.add_argument("--JustEnterROI", help="select tracks that cross multiple edges of the roi", action="store_true")
    parser.add_argument("--JustExitROI", help="select tracks that cross multiple edges of the roi", action="store_true")
    parser.add_argument("--WithinROI", help="select tracks that cross multiple edges of the roi", action="store_true")
    parser.add_argument("--ExitOrCrossROI", help="select tracks that either exit or cross multi roi", action="store_true")
    parser.add_argument("--SelectToBeCounted", help="select tracks to be counted based on ROI", action="store_true")
    parser.add_argument("--UnfinishedTrackFrameTh", help="select tracks to be counted based on ROI", type=int, default=10)
    parser.add_argument("--UnstartedTrackFrameTh", help="select tracks to be counted based on ROI", type=int, default=10)

    parser.add_argument("--MaskGPFrame", help="remove dets on tracks that are outside gp frame", action="store_true")
    parser.add_argument("--TrackEval", help="Evaluate the tracking single camera", action="store_true")
    parser.add_argument("--VisROI", help="visualize the selected ROI", action='store_true')
    parser.add_argument("--VisTrackMoI", help="visualize tracking with moi labels", action='store_true')
    parser.add_argument("--LabelledTrajectories", help=" a pkl file containint the labelled trajectories on the ground plane",type=str)
    parser.add_argument("--ExtractCommonTracks", help="instead of track labelling GUI extract common tracks automatically", action='store_true')
    parser.add_argument("--UseCachedCounter", help="use a pre-initialized cached counter object", action='store_true')
    parser.add_argument("--CacheCounter", help="Cache the counter after initialization", action='store_true')
    parser.add_argument("--VisTrackTop", help="Visualize tracks from top view", action='store_true')
    parser.add_argument("--CachedCounterPth", help="path to pre-initialized cached counter object", type=str)
    parser.add_argument("--StartFrame", help="For if the video hasnt been cropped to the start frame yet, to be removed soon", type=int)
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
    parser.add_argument("--ContactPoint", help="Select how to set the contact point  options = [BottomPoint/Center/BottomSeg/SegBottomLine]", type=str)
    parser.add_argument("--VisContactPoint", help="to visualize the contact point", action="store_true")
    parser.add_argument("--VisCPTop", help="to visualize the contact points on the top view", action="store_true")

    parser.add_argument("--Segment", help="perform segmentation and store results", action='store_true')
    parser.add_argument("--VisSegment", help="Vis Segmentation Masks", action='store_true')
    parser.add_argument("--Segmenter", help="model for segmentation", type=str, default="Null")
    parser.add_argument("--SegTh", help="threshold to filter segmentation masks", type=float)
    parser.add_argument("--SegPostProc", help="perform segmentation post processing", action='store_true')
    parser.add_argument("--ExtractImages", help="extract images from video and store under results/Images", action='store_true')
    parser.add_argument("--ConvertDetsToCOCO", help="convert detection files to COCO format", action='store_true')
    parser.add_argument("--KeepCOCOClasses", help="when converting to COCO keep coco class names", action='store_true')

    parser.add_argument("--MultiSeg", help="operating on multiple segments(eg train segments)", action='store_true')
    parser.add_argument("--MultiPart", help="for multi part operations", action='store_true')
    parser.add_argument("--FineTune", help="fine tune detector", action='store_true')
    parser.add_argument("--Resume", help="resume fine tunning detectors", action='store_true')
    parser.add_argument("--TrainPart", help="training SubID", type=str)
    parser.add_argument("--ValidPart", help="validation SubID", type=str)
    parser.add_argument("--GTDetector", help="name of GT detector(typically used for fine turning or evaluation)", type=str)
    parser.add_argument("--GTDetector3D", help="name of 3D GT detector", type=str)
    parser.add_argument("--GTTracker3D",  help="name of 3D GT tracker", type=str)
    parser.add_argument("--GTTracker",  help="name of GT tracker(typically used for fine turning or evaluation)", type=str)
    parser.add_argument("--BatchSize", help="set batch size", type=int)
    parser.add_argument("--NumWorkers", help="number of workers for dataloader", type=int)
    parser.add_argument("--Epochs", help="number of epochs", type=int)
    parser.add_argument("--ValInterval", help="frequency of validation step(every x epochs)", type=int)
    parser.add_argument("--CalcDistance", help="Calculate distance from tracking result to intersection", action="store_true")
    parser.add_argument("--SAHI", help="run detection with SAHI", action='store_true')
    parser.add_argument("--SahiPatchSize", help="patch size of sahi", type=int, default=640)
    parser.add_argument("--SahiPatchOverlapRatio", help="overlap ration of sahi patches", type=float, default=0.25)
    parser.add_argument("--SahiPatchBatchSize", help="batch size of patches of sahi", type=int, default=0)
    parser.add_argument("--SahiNMSTh", help="IoU threshould for merging results when using sahi", type=float, default=0.25)

    parser.add_argument("--CVPR", help="prepare CVPR stats of dataset", action='store_true')
    parser.add_argument("--OSR", help="over sampling reatio after resampling", type=int, default=10)
    parser.add_argument("--KDEBW", help="BandWidth of KDE", type=float)
    parser.add_argument("--GP", help="operate on ground plane", action='store_true')
    parser.add_argument("--ROIFromTop", help="get roi from topview need to select topview", action='store_true', default=False)
    parser.add_argument("--UnifyTrackClass", help="unify class labels for each track", action='store_true')

    parser.add_argument("--EvalContactPoitnSelection", help="evaluate contact point selection method using 2D-3D GT", action='store_true')

    return parser
    
if __name__ == "__main__":
    # ferch the arguments
    parser = get_parser()
    args = parser.parse_args()

    # check if the opeerations should be performed cross camera
    if args.MultiCam:
        args , args_mc = get_args_mc(args)
        main_mc(args, args_mc)

    elif args.MultiSeg:
        args, args_ms, args_mcs = get_args_ms(args)
        main_ms(args, args_ms, args_mcs)

    elif args.MultiPart:
        args, args_mp, args_mss, args_mcs = get_args_mp(args)
        main_mp(args, args_mp, args_mss, args_mcs)

    else:
        args = get_args(args)
        main(args)