import sys
import pickle as pkl
import json
# import argparse
import sys
CENTERTRACK_PATH = "./Trackers/CenterTrack/CenterTrack/src/lib/"
sys.path.insert(0, CENTERTRACK_PATH)
# import cv2
# from tqdm import tqdm
# import torch

if __name__ == "__main__":
    args = json.loads(sys.argv[-1]) # args in a dictionary here where it was a argparse.NameSpace in the main code
    print(args["DetectionDetectorPath"])    
