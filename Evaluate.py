from Utils import *
from Libs import *

def get_offset_frames(dfs):
    # max_frames = [np.max(df[["fn"]].to_numpy()) for df in dfs]
    max_frames = [100*len(df) for df in dfs]

    frame_offset = [0]
    for mx in max_frames[:-1]:
        frame_offset.append(mx)
    return frame_offset

def combine_dfs(dfs, gts):
    frame_offset = get_offset_frames(gts)
    temp = 0
    for i, off_frames in enumerate(frame_offset):
        temp += len(dfs[i])
        dfs[i][["fn"]] = dfs[i][["fn"]] + off_frames

    return pd.concat(dfs)

def compare_dataframes(gts, ts):
    """Builds accumulator for each sequence."""
    accs = []
    names = []
    for k, tsacc in ts.items():
        accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
        names.append(k)
    return accs, names

def prepare_df_for_motmetric(dfs, cam_ids):
    '''
    in this df we assume that df has the following keys
    x1, y1, x2, y2, id, fn
    '''
    labels = defaultdict(list)
    for df, cam_id in zip(dfs, cam_ids):
        raw_list = []
        for i, row in df.iterrows():
            raw_list.append({'FrameId':row.fn, 'Id':int(row.id), 'X':int(float(row.x1)), 'Y':int(float(row.y1)), 'Width':int(float(row.x2-row.x1)), 'Height':int(float(row.y2-row.y1)), 'Confidence':1.0})
        labels[cam_id].extend(raw_list)
    return OrderedDict([(cam_id, pd.DataFrame(rows).set_index(['FrameId', 'Id'])) for cam_id, rows in labels.items()])


def evaluate_tracking(base_args, nested_args):
    '''
    single source evaluation of tracking
    nested_args: a nested list of arg namespaces. Each arg is for one video
    '''
    dfs_pred = []
    dfs_gt   = []
    dfs_ids  = []
    
    flat_args = flatten_args(nested_args)
    for args in flat_args:
        args_gt = get_args_gt(args)

        df_gt   = pd.read_pickle(args_gt.TrackingPkl)
        df_pred = pd.read_pickle(args.TrackingPkl)
        df_id   = args.SubID

        dfs_gt.append(df_gt)
        dfs_pred.append(df_pred)
        dfs_ids.append(df_id)
    
    # prepare dfs for mot metrics(transfer from local format to mot format)
    gt_dfs = prepare_df_for_motmetric(dfs_gt, dfs_ids)
    pred_dfs = prepare_df_for_motmetric(dfs_pred, dfs_ids)

    accs, names = compare_dataframes(gt_dfs, pred_dfs)
    mh = mm.metrics.create()
    summary = mh.compute_many(accs, names=names, metrics=mm.metrics.motchallenge_metrics, generate_overall=True)

    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)


    # dfs_pred =[]
    # dfs_gt   =[]
    # cam_ids  =[]
    # for arg in args_mc:
    #     df = pd.read_pickle(arg.TrackingPkl)
    #     dfs_pred.append(df)
    #     cam_ids.append(arg.CamID)
    
    # for arg in args_mc_gt:
    #     df = pd.read_pickle(arg.TrackingPkl)
    #     dfs_gt.append(df)

    # for a,arg in enumerate(args_mc):
    #     df = pd.read_pickle(arg.TrackingPkl)
    #     g=(dfs_gt[a])
    #     max_fn=g.max(axis=0)['fn']
    #     min_fn=g.min(axis=0)['fn']
    #     df=df[(df['fn'] >= min_fn) & (df['fn'] <=max_fn ) &((df['fn']-min_fn)%6==0)]
    #     dfs_pred.append(df)
    #     cam_ids.append(arg.CamID)

    # df_combined_pred = combine_dfs(dfs_pred, dfs_gt)
    # df_combined_gt   = combine_dfs(dfs_gt, dfs_gt)
    # gt_dfs = prepare_df_for_motmetric([df_combined_gt], ["MC"])
    # pred_dfs = prepare_df_for_motmetric([df_combined_pred], ["MC"])

    # accs, names = compare_dataframes(gt_dfs, pred_dfs)
    # mh = mm.metrics.create()
    # summary = mh.compute_many(accs, names=names, metrics=mm.metrics.motchallenge_metrics)

    # strsummary_mc = mm.io.render_summary(
    #     summary,
    #     formatters=mh.formatters,
    #     namemap=mm.io.motchallenge_metric_names
    # )
    # print(strsummary_mc)



    with open(base_args.TrackEvalPth, "w") as f:
        f.write(strsummary)
        # f.write("\n")
        # f.write(strsummary_mc)

    return SucLog("Track Evaluation Successful")

import seaborn as sns
# import matplotlib
# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': True,
# })
plt.style.use("ggplot")
import matplotlib as mpl
plt.close()
mpl.rcParams.update(mpl.rcParamsDefault)
import tikzplotlib

def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)

def cvpr(base_args, nested_args):
    dfs_train = []
    dfs_train_ids  = []

    dfs_valid = []
    dfs_valid_ids  = []

    train_nested_args = nested_args[0]
    valid_nested_args = nested_args[1]

    flat_args_train = flatten_args(train_nested_args)
    flat_args_valid = flatten_args(valid_nested_args)

    for args in flat_args_train:
        df    = pd.read_pickle(args.DetectionPkl)
        df_id = args.SubID
        dfs_train.append(df)
        dfs_train_ids.append(df_id)

    for args in flat_args_valid:
        df    = pd.read_pickle(args.DetectionPkl)
        df_id = args.SubID
        dfs_valid.append(df)
        dfs_valid_ids.append(df_id)

    df_train = pd.concat(dfs_train)
    df_valid = pd.concat(dfs_valid)

    if base_args.Detector == "GTHW7FG":
        print(df_train.label.unique())
        print(df_train["class"].unique())

    if base_args.Detector == "GTHW73D":

        df_train.loc[df_train.label_fg == 'Articulated Municipal Transit Buses', "label_fg"] = "AM"
        df_train.loc[df_train.label_fg == 'Single-Unit Municipal Transit Buses', "label_fg"] = "SM"
        df_train.loc[df_train.label_fg == 'Pickup Truck', "label_fg"] = "PT"
        df_train.loc[df_train.label_fg == 'Light Trucks', "label_fg"] = "LT"
        df_train.loc[df_train.label_fg == 'Medium Trucks', "label_fg"] = "MT"
        df_train.loc[df_train.label_fg == 'Tractor-Trailer', "label_fg"] = "TT"
        df_train.loc[df_train.label_fg == 'Heavy Trucks', "label_fg"] = "HT"
        df_train.loc[df_train.label_fg == 'Tractor Only', "label_fg"] = "TO"
        df_train.loc[df_train.label_fg == 'Minivan', "label_fg"] = "MV"
        df_train.loc[df_train.label_fg == 'Pedestrian', "label_fg"] = "P"
        df_train.loc[df_train.label_fg == 'Unpowered', "label_fg"] = "UP"

        df_valid.loc[df_valid.label_fg == 'Articulated Municipal Transit Buses', "label_fg"] = "AM"
        df_valid.loc[df_valid.label_fg == 'Single-Unit Municipal Transit Buses', "label_fg"] = "SM"
        df_valid.loc[df_valid.label_fg == 'Pickup Truck', "label_fg"] = "PT"
        df_valid.loc[df_valid.label_fg == 'Light Trucks', "label_fg"] = "LT"
        df_valid.loc[df_valid.label_fg == 'Medium Trucks', "label_fg"] = "MT"
        df_valid.loc[df_valid.label_fg == 'Tractor-Trailer', "label_fg"] = "TT"
        df_valid.loc[df_valid.label_fg == 'Heavy Trucks', "label_fg"] = "HT"
        df_valid.loc[df_valid.label_fg == 'Tractor Only', "label_fg"] = "TO"
        df_valid.loc[df_valid.label_fg == 'Minivan', "label_fg"] = "MV"
        df_valid.loc[df_valid.label_fg == 'Pedestrian', "label_fg"] = "P"
        df_valid.loc[df_valid.label_fg == 'Unpowered', "label_fg"] = "UP"


        df_train.loc[df_train.label_cg == 'Small Vehicles', "label_cg"] = "SV"
        df_train.loc[df_train.label_cg == 'Single-unit Trucks', "label_cg"] = "ST"
        df_train.loc[df_train.label_cg == 'Buses', "label_cg"] = "B"
        df_train.loc[df_train.label_cg == 'Articulated Trucks', "label_cg"] = "AT"
        df_train.loc[df_train.label_cg == 'Pedestrian' , "label_cg"] = "P"
        df_train.loc[df_train.label_cg == 'Two-Wheelers' , "label_cg"] = "TW"


        df_valid.loc[df_valid.label_cg == 'Small Vehicles', "label_cg"] = "SV"
        df_valid.loc[df_valid.label_cg == 'Single-unit Trucks', "label_cg"] = "ST"
        df_valid.loc[df_valid.label_cg == 'Buses', "label_cg"] = "B"
        df_valid.loc[df_valid.label_cg == 'Articulated Trucks', "label_cg"] = "AT"
        df_valid.loc[df_valid.label_cg == 'Pedestrian' , "label_cg"] = "P"
        df_valid.loc[df_valid.label_cg == 'Two-Wheelers' , "label_cg"] = "TW"

    
        # plt.figure(figsize=(12, 9))
        # sns.kdeplot(df_train["x_dim"], shade=True, label="train", color="blue")
        # sns.kdeplot(df_valid["x_dim"], shade=True, label="valid", color="green")
        # plt.legend()
        # plt.xlabel("Width [m]")
        # fig = plt.gcf()
        # tikzplotlib_fix_ncols(fig)
        # tikzplotlib.save("Sup_Width.tex")
        # plt.savefig("Sup_Width.pdf")
        # plt.close("all")

        # plt.figure(figsize=(12, 9))
        # sns.kdeplot(df_train["y_dim"], shade=True, label="train", color="blue")
        # sns.kdeplot(df_valid["y_dim"], shade=True, label="valid", color="green")
        # plt.legend()
        # plt.xlabel("Length [m]")
        # fig = plt.gcf()
        # tikzplotlib_fix_ncols(fig)
        # tikzplotlib.save("Sup_Length.tex")
        # plt.savefig("Sup_Length.pdf")
        # plt.close("all")

        # plt.figure(figsize=(12, 9))
        # sns.kdeplot(df_train["z_dim"], shade=True, label="train", color="blue")
        # sns.kdeplot(df_valid["z_dim"], shade=True, label="valid", color="green")
        # plt.legend()
        # plt.xlabel("Height [m]")
        # fig = plt.gcf()
        # tikzplotlib_fix_ncols(fig)
        # tikzplotlib.save("Sup_Height.tex")
        # plt.savefig("Sup_Height.pdf")
        # plt.close("all")

        # df_train["yaw"] = df_train["yaw"] * 180.0 /np.pi
        # df_valid["yaw"] = df_valid["yaw"] * 180.0 /np.pi
        # plt.figure(figsize=(12, 9))
        # sns.kdeplot(df_train["yaw"], shade=True, label="train", color="blue")
        # sns.kdeplot(df_valid["yaw"], shade=True, label="valid", color="green")
        # plt.legend()
        # plt.xlabel("Yaw [deg]")
        # plt.xlim(-180, +180)
        # fig = plt.gcf()
        # tikzplotlib_fix_ncols(fig)
        # tikzplotlib.save("Sup_Yaw.tex")
        # plt.savefig("Sup_Yaw.pdf")
        # plt.close("all")

        # df_train["roll"] = df_train["roll"] * 180.0 / np.pi
        # df_valid["roll"] = df_valid["roll"] * 180.0 / np.pi
        # plt.figure(figsize=(12, 9))
        # sns.kdeplot(df_train["roll"], shade=True, label="train", color="blue")
        # sns.kdeplot(df_valid["roll"], shade=True, label="valid", color="green")
        # plt.legend()
        # plt.xlabel("Roll [deg]")
        # plt.xlim(-1, +1)
        # fig = plt.gcf()
        # tikzplotlib_fix_ncols(fig)
        # tikzplotlib.save("Sup_Roll.tex")
        # plt.savefig("Sup_Roll.pdf")
        # plt.close("all")


        # df_train["pitch"] = df_train["pitch"] * 180.0 /np.pi    
        # df_valid["pitch"] = df_valid["pitch"] * 180.0 /np.pi
        # plt.figure(figsize=(12, 9))
        # sns.kdeplot(df_train["pitch"], shade=True, label="train", color="blue")
        # sns.kdeplot(df_valid["pitch"], shade=True, label="valid", color="green")
        # plt.legend()
        # plt.xlabel("Pitch [deg]")
        # plt.xlim(-1, +1)
        # fig = plt.gcf()
        # tikzplotlib_fix_ncols(fig)
        # tikzplotlib.save("Sup_Pitch.tex")
        # plt.savefig("Sup_Pitch.pdf")
        # plt.close("all")


        # df_train["split"] = ["train" for _ in range(len(df_train))]
        # df_valid["split"] = ["valid" for _ in range(len(df_valid))]
        # df_merged = pd.concat((df_train, df_valid))
        # plt.figure(figsize=(12, 9))

        # # sns.countplot(df_train, x="label_fg", label="train", color="blue", alpha=0.5 , dodge=True, stat="probability")
        # # sns.countplot(df_valid, x="label_fg", label="valid", color="green", alpha=0.5, dodge=True, stat="probability")
        # b = sns.histplot(data=df_merged, x="label_fg",  alpha=0.5 , multiple="dodge", stat="probability",
        #     hue="split", palette={"train":"blue", "valid":"green"}, shrink=.8, legend=True, common_norm=False)

        # # b.tick_params(labelsize=20)
        # # plt.legend()
        # plt.xlabel("Fine-Grained Class Labels")
        # plt.xticks(fontsize=15)
        # fig = plt.gcf()
        # tikzplotlib_fix_ncols(fig)
        # tikzplotlib.save("Sup_FGLabel.tex")
        # plt.savefig("Sup_FGLabel.pdf")
        # plt.close("all")
        # print(df_train["label_fg"].unique())


        # df_train["split"] = ["train" for _ in range(len(df_train))]
        # df_valid["split"] = ["valid" for _ in range(len(df_valid))]
        # df_merged = pd.concat((df_train, df_valid))
        # plt.figure(figsize=(12, 9))
        # # sns.countplot(df_train, y="label_cg", label="train", color="blue", alpha=0.5 , dodge=True)
        # # sns.countplot(df_valid, y="label_cg", label="valid", color="green", alpha=0.5, dodge=True)
        # sns.histplot(data=df_merged, x="label_cg",  alpha=0.5 , multiple="dodge", stat="probability",
        #     hue="split", palette={"train":"blue", "valid":"green"}, shrink=.8, legend=True, common_norm=False)
        # # plt.legend()
        # plt.xlabel("Coarse-Grained Class Labels")
        # fig = plt.gcf()
        # tikzplotlib_fix_ncols(fig)
        # tikzplotlib.save("Sup_CGLabel.tex")
        # plt.savefig("Sup_CGLabel.pdf")
        # plt.close("all")
        # print(df_train["label_cg"].unique())

        # plt.figure(figsize=(12, 9))
        # sns.kdeplot(df_train["numberOfPoints"], shade=True, label="train", color="blue")
        # sns.kdeplot(df_valid["numberOfPoints"], shade=True, label="valid", color="green")
        # plt.legend()
        # plt.xlabel("Mumber of LiDAR Points in Cuboid")
        # plt.xlim(0, 1000)
        # fig = plt.gcf()
        # tikzplotlib_fix_ncols(fig)
        # tikzplotlib.save("Sup_numberOfPoints.tex")
        # plt.savefig("Sup_numberOfPoints.pdf")
        # plt.close("all")

        # plt.figure(figsize=(12, 9))
        # sns.kdeplot(df_train['distance_to_device'], shade=True, label="train", color="blue")
        # sns.kdeplot(df_valid['distance_to_device'], shade=True, label="valid", color="green")
        # plt.legend()
        # plt.xlabel("Distance from LiDAR Sensor[m]")
        # fig = plt.gcf()
        # tikzplotlib_fix_ncols(fig)
        # tikzplotlib.save("Sup_distance.tex")
        # plt.savefig("Sup_distance.pdf")
        # plt.close("all")


        # df_train.loc[df_train["motion_state"] == "UNK", "motion_state"] = "Unknown"
        # df_valid.loc[df_valid["motion_state"] == "UNK", "motion_state"] = "Unknown"

        # df_train.loc[df_train["motion_state"] == "Parked", "motion_state"] = "Stopped"
        # df_valid.loc[df_valid["motion_state"] == "Parked", "motion_state"] = "Stopped"
        

        # df_train["split"] = ["train" for _ in range(len(df_train))]
        # df_valid["split"] = ["valid" for _ in range(len(df_valid))]
        # df_merged = pd.concat((df_train, df_valid))
        # df_merged = df_merged.drop(df_merged[df_merged['motion_state'] == 'Unknown'].index)

        
        # plt.figure(figsize=(12, 9))

        # sns.histplot(data=df_merged, x="motion_state",  alpha=0.5 , multiple="dodge", stat="probability",
        # hue="split", palette={"train":"blue", "valid":"green"}, shrink=.8, legend=True, common_norm=False)
        # sns.countplot(df_train, x="motion_state", label="train", color="blue",  alpha=0.5 , dodge=True, stat="probability")
        # sns.countplot(df_valid, x="motion_state", label="valid", color="green", alpha=0.5, dodge=True , stat="probability")

        # plt.legend()
        # plt.ylim(0, 1)
        # plt.xlabel("Motion State")
        # fig = plt.gcf()
        # tikzplotlib_fix_ncols(fig)
        # tikzplotlib.save("Sup_MotionState.tex")
        # plt.savefig("Sup_MotionState.pdf")
        # plt.close("all")

        # df_train.loc[df_train["traversal_direction"]=="UNK", "traversal_direction"] = "Unknown"
        # df_valid.loc[df_valid["traversal_direction"]=="UNK", "traversal_direction"] = "Unknown"

        # df_train.loc[df_train["traversal_direction"]=="Straight-Through", "traversal_direction"] = "Straight"
        # df_valid.loc[df_valid["traversal_direction"]=="Straight-Through", "traversal_direction"] = "Straight"

        # df_train["split"] = ["train" for _ in range(len(df_train))]
        # df_valid["split"] = ["valid" for _ in range(len(df_valid))]
        # df_merged = pd.concat((df_train, df_valid))

        # df_merged = df_merged.drop(df_merged[df_merged['traversal_direction'] == 'Unknown'].index)

        # plt.figure(figsize=(12, 9))
        # sns.histplot(data=df_merged, x="traversal_direction",  alpha=0.5 , multiple="dodge", stat="probability",
        # hue="split", palette={"train":"blue", "valid":"green"}, shrink=.8, legend=True, common_norm=False)

        # # sns.countplot(df_train, x="traversal_direction", label="train", color="blue", alpha=0.5 , dodge=True)
        # # sns.countplot(df_valid, x="traversal_direction", label="valid", color="green", alpha=0.5, dodge=True)
        # # plt.legend()
        # plt.xlabel("Intersection Traversal Direction")
        # plt.savefig("Sup_traversal_direction.pdf")
        # fig = plt.gcf()
        # tikzplotlib_fix_ncols(fig)
        # tikzplotlib.save("Sup_traversal_direction.tex")
        # plt.close("all")