from Libs import *
from Utils import *
from Detectors.MMDet.detect import setup as mm_setup
from Detectors.MMDet.detect import df as mm_df
from Detectors.MMDet.detect import df_txt as mm_df_txt

def detect(args, *oargs):
    setup(args)
    env_name = "MMDet"
    exec_path = "./Detectors/MMDet/run.py"
    config_file = "./Detectors/MMDet/mmdetection/configs/rtmdet/rtmdet_x_8xb32-300e_coco.py"
    checkpoint_file = "./Detectors/MMDet/mmdetection/checkpoints/rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth"
    args.MMDetConfig = config_file 
    args.MMDetCheckPoint = checkpoint_file
    conda_pyrun(env_name, exec_path, args)

def df(args):
    return mm_df(args)

def df_txt(df,text_result_path):
    return mm_df_txt(df,text_result_path)

def setup(args):
    mm_setup(args)
    # download checkpoints
    try:
        os.system("mkdir ./Detectors/MMDet/mmdetection/checkpoints")
    except:
        pass
    # download weights
    download_url_to("https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_x_8xb32-300e_coco/rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth",
                     "./Detectors/MMDet/mmdetection/checkpoints/rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth")