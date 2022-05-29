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
from Detect import detect, visdetect
from Track import track, vistrack
from Homography import homographygui
from Homography import reproject

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

def Homography(args):
    if args.Homography:
        print(ProcLog("Homography reprojection in Process"))
        log = reproject(args)
        return log
    else: return WarningLog("skipped homography subtask")

def TrackLabelingGUI(args):
    if args.TrackLabelingGUI:
        raise NotImplemented
    else: return WarningLog("skipped track labelling subtask")

def Count(args):
    if args.Count:
        raise NotImplemented
    else: return WarningLog("skipped counting subtask")

def main(args):
    # Pass the args to each subtask
    # Each subtask will validate its own inputs
    subtasks = [Preprocess, Detect, VisDetect, Track, VisTrack, HomographyGUI, Homography, TrackLabelingGUI, Count]
    for subtask in subtasks:
        log = subtask(args)
        print(log)
    
if __name__ == "__main__":
    # ferch the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--Dataset", help="Path to a rep containing video files", type=str)
    parser.add_argument("--Video", help="a list of video pathes in one repo", type=list)
    parser.add_argument("--Preprocess", help="If preprocess inputs first", action="store_true")
    parser.add_argument("--Detect", help="If perform detection", action="store_true")
    parser.add_argument("--VisDetect", help="If create video of detections", action="store_true")
    parser.add_argument("--Track", help="If perform tracking", action="store_true")
    parser.add_argument("--VisTrack", help="If create video visualization of tracking", action="store_true")
    parser.add_argument("--HomographyGUI", help="If pop-up homography GUI", action="store_true")
    parser.add_argument("--TrackLabelingGUI", help="If pop-up Track Labeling GUI", action="store_true")
    parser.add_argument("--Homography", help="If perform backkprojection using homography matrix", action="store_true")
    parser.add_argument("--Count", help="If count the objects for each MOI", action="store_true")
    parser.add_argument("--Detector", help="Name of detector to be used", type=str)
    parser.add_argument("--Tracker", help="Name of tracker to be used", type=str)

    args = parser.parse_args()

    args = complete_args(args)
    check_config(args)

    main(args)
