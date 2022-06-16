# this file is developed to envoce tracklabeling GUI
from Utils import * 
from Libs import *

# args variables for tracklabelling gui
    # args.TrackLabellingExportPth

def tracklabelinggui(args):
    export_path = os.path.abspath(args.TrackLabellingExportPth)
    cwd = os.getcwd()
    os.chdir("./track_labelling_gui/")
    ret = os.system(f"sudo python3 cam_gen.py --Export='{export_path}'")
    os.chdir(cwd)

    if ret==0:
        return SucLog("track labelling executed successfully")
    return FailLog("track labelling ended with non-zero return value")


def vislabelledtracks(args):
    raise NotImplemented
