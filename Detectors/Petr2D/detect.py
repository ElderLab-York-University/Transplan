from Libs import *
from Utils import *
from Detectors.MMDet3D.detect import setup as mm_setup
from Detectors.MMDet3D.detect import df as mm_df
from Detectors.MMDet3D.detect import df_txt as mm_df_txt
# from Detectors.MMDet3D.detect import fine_tune as mm_fine_tune
# from Detectors.MMDet3D.detect import modify_train_config as mm_modify_train_config

def detect(args, *oargs):
    env_name        = "MMDet3D"
    exec_path       = "./Detectors/MMDet3D/run.py"
    config_file     = config_from_version(args.DetectorVersion)
    checkpoint_file = checkpoint_from_version(args.DetectorVersion)

    setup(args)
    args.MMDetConfig = config_file 
    args.MMDetCheckPoint = checkpoint_file
    conda_pyrun(env_name, exec_path, args)

def df(args):
    return mm_df(args)

def df_txt(df,text_result_path):
    return mm_df_txt(df,text_result_path)

def setup(args):
    mm_setup(args)


# def fine_tune(args, args_mp, args_gt, args_mp_gt):
#     # modify config file with args
#     train_config_path = "./Detectors/YoloX/train_config.py"
#     mm_modify_train_config(train_config_path, args, args_mp, args_gt, args_mp_gt)

#     work_dir = args.DetectorCheckPointDir
#     if not os.path.isdir(work_dir):
#         os.system(f"mkdir {work_dir}")

#     mm_fine_tune(train_config_path, work_dir, args.Resume)

def checkpoint_from_version(version):
    c_2_v = {
        "kitti"      : "./CheckPoints/petr_vovnet_gridmask_p4_800x320-e2191752.pth",
    }
    return c_2_v[version]

def config_from_version(version):
    c_2_v = {
        "kitti"      : "./CheckPoints/petr_vovnet_gridmask_p4_800x320.py",
    }
    return c_2_v[version]