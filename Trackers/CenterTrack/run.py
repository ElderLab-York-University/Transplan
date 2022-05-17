print("I am here in the run.py file; hopefully this work for you")
import sys
import pickle as pkl
import json
import argparse
import sys
CENTERTRACK_PATH = "./Trackers/CenterTrack/CenterTrack/src/lib/"
sys.path.insert(0, CENTERTRACK_PATH)
from detector import Detector
from opts import opts

if __name__ == "__main__":
    print("in run.py main part")
    args = json.loads(sys.argv[-1]) # args in a dictionary here where it was a argparse.NameSpace in the main code
    # print(args.keys())

    MODEL_PATH = "./Trackers/CenterTrack/CenterTrack/models/coco_tracking.pth"
    TASK = 'tracking' # or 'tracking,multi_pose' for pose tracking and 'tracking,ddd' for monocular 3d tracking
    opt = opts().init('{} --load_model {}'.format(TASK, MODEL_PATH).split(' '))
    print("opt")
    print(opt)
    detector = Detector(opt)
    print("end of run.py reached")

    # images = ['''image read from open cv or from a video''']
    # for img in images:
    # ret = detector.run(img)['results']

    
