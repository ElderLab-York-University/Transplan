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
        if(args.ForNFrames is not None):
            df_gt=df_gt[df_gt['fn']< args.ForNFrames]
            df_pred=df_pred[df_pred['fn']< args.ForNFrames]
            
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
# import tikzplotlib

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


    df_train.loc[df_train["motion_state"] == "UNK", "motion_state"] = "Unknown"
    df_valid.loc[df_valid["motion_state"] == "UNK", "motion_state"] = "Unknown"

    df_train.loc[df_train["motion_state"] == "Parked", "motion_state"] = "Stopped"
    df_valid.loc[df_valid["motion_state"] == "Parked", "motion_state"] = "Stopped"
    

    df_train["split"] = ["train" for _ in range(len(df_train))]
    df_valid["split"] = ["valid" for _ in range(len(df_valid))]
    df_merged = pd.concat((df_train, df_valid))
    df_merged = df_merged.drop(df_merged[df_merged['motion_state'] == 'Unknown'].index)

    
    plt.figure(figsize=(12, 9))

    sns.histplot(data=df_merged, x="motion_state",  alpha=0.5 , multiple="dodge", stat="probability",
    hue="split", palette={"train":"blue", "valid":"green"}, shrink=.8, legend=True, common_norm=False)
    # sns.countplot(df_train, x="motion_state", label="train", color="blue",  alpha=0.5 , dodge=True, stat="probability")
    # sns.countplot(df_valid, x="motion_state", label="valid", color="green", alpha=0.5, dodge=True , stat="probability")

    # plt.legend()
    plt.ylim(0, 1)
    plt.xlabel("Motion State")
    fig = plt.gcf()
    tikzplotlib_fix_ncols(fig)
    tikzplotlib.save("Sup_MotionState.tex")
    plt.savefig("Sup_MotionState.pdf")
    plt.close("all")

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


def evaluate_detection(base_args, nested_args):
    dfs_pred = []
    dfs_gt   = []
    dfs_ids  = []
    
    flat_args = flatten_args(nested_args)
    # classes=[]
    # for args in flat_args:
    #     args_gt = get_args_gt(args)
    #     df_gt   = pd.read_pickle(args_gt.DetectionPkl)
    #     df_pred = pd.read_pickle(args.DetectionPklBackUp)
    #     n_classes=np.unique(df_pred['class'])
    #     gt_frames=np.unique(df_gt['fn'])
    #     df_pred=df_pred[df_pred['fn'].isin(gt_frames)]
        
    #     for c in n_classes:
    #         if c not in classes:
    #             classes.append(c)
    # classes=np.sort(classes)        
    # class_dict={}
    # for c in range(len(classes)):
    #     class_dict[classes[c]]=c
    # det_results=[]
    # annotations=[]    
    class_dict={0:0,1:1,2:2,3:5,4:7,7:7}
    print(class_dict)
    preds=[]
    gts=[]
    q=0
    
    print('starting')
    if(base_args.Small):
        print(base_args.DetectEvalPthSmall)
    if(base_args.Medium):
        print(base_args.DetectEvalPthMedium)
    if(base_args.Large):
        print(base_args.DetectEvalPthLarge)
        
    for args in flat_args:
        camera=args.Dataset[-3:]
        if(args.Camera is None or camera in args.Camera):
            args_gt = get_args_gt(args)
                
            df_gt   = pd.read_pickle(args_gt.DetectionPkl)
            # if(os.path.exists(args.DetectionPkl)):
            print(args.Dataset)
            if(args.Rois is not None):
                df_pred=pd.read_pickle(args.DetectionPkl)
            else:
                df_pred = pd.read_pickle(args.DetectionPklBackUp)
            df_id   = args.SubID
            gt_frames=np.unique(df_gt['fn'])
            df_pred=df_pred[df_pred['fn'].isin(gt_frames)]
            # print(np.unique(df_pred['class']))            
            # df_pred=df_pred.replace({"class":class_dict})
            # print(np.unique(df_pred['class']))
            # print(df_gt)
            # print(args_gt.DetectionPkl)
            # input()
            df_gt['df_id']=q
            df_pred['df_id']=q
            q=q+1
            # for fn in np.unique(df_gt['fn']):
            #     gt_frame=np.asarray(df_gt[df_gt['fn']==fn])
            #     frame_gt=[]
            #     for gt in gt_frame:
            #         frame_gt.append([gt[1],gt[3],gt[4],gt[5],gt[6]])
            #     pred_frame=np.asarray(df_pred[df_pred['fn']==fn])
            #     frame_pred=[]
            #     for pred in pred_frame:
            #         frame_pred.append([pred[1],pred[2],pred[3],pred[4],pred[5],pred[6]])
            #     preds.append(frame_pred)
            #     gts.append(frame_gt)
                
                
            # for fn in np.unique(
                # df_gt['fn']):
            #     frame_list=[]
            #     preds_frame=(df_pred[df_pred['fn']==fn])    
            #     for c in classes: 
            #         preds_classes=np.asarray(preds_frame[preds_frame['class']==c])[:,3:-1]
            #         frame_list.append(preds_classes)
            #     det_results.append(frame_list)
            # for fn in np.unique(df_gt['fn']):
            #     frame_dict={}
            #     gts_frame=(df_gt[df_gt['fn']==fn])    
            #     gts_classes=np.asarray(gts_frame)[:,3:-1]
            #     gts_ls=np.asarray(gts_frame)[:,1]
            #     gts_labels=np.zeros(gts_ls.shape)
            #     for i in range(len(gts_ls)):
            #         if(gts_ls[i] in class_dict):
            #             gts_labels[i]=class_dict[gts_ls[i]]
            #         else:
            #             gts_labels[i]=gts_ls[i]                    
                        
            #     frame_dict['bboxes']=gts_classes
            #     frame_dict['labels']=gts_labels
                
            #     annotations.append(frame_dict)
            if args.Intersection:
                if args.Dataset[-3:-1]=='sc':
                    roi=args.MetaData['roi']
                    df_gt,_=remove_out_of_ROI(df_gt,roi)
                    df_pred,_=remove_out_of_ROI(df_pred,roi)  
                    dfs_gt.append(df_gt)
                    dfs_pred.append(df_pred)
                                
            elif args.NotIntersection:
                if args.Dataset[-3:-1]=='sc':
                    roi=args.MetaData['roi']
                    _,df_gt=remove_out_of_ROI(df_gt,roi)
                    _,df_pred=remove_out_of_ROI(df_pred,roi)  
                    dfs_gt.append(df_gt)
                    dfs_pred.append(df_pred)
            else:
                dfs_gt.append(df_gt)
                dfs_pred.append(df_pred)
                dfs_ids.append(df_id)
    gts=np.array(gts)
    preds=np.array(preds)
    dfs_gt=pd.concat(dfs_gt)
    dfs_pred=pd.concat(dfs_pred)
    classes=np.unique(dfs_pred['class'])
    print(classes)
    print(np.unique(dfs_gt['class']))
    # dfs_pred=dfs_pred[dfs_pred['class']==1]
    # dfs_gt=dfs_gt[dfs_pred['class']==1]    
    # dfs_pred['class']= dfs_pred['class'].map(class_dict)
    
    
    classes=np.unique(dfs_pred['class'])
    print(classes)
    print('Starting')
    # with open("det_results", 'wb') as f:
    #     pkl.dump(det_results, f)
    # with open("annotations", 'wb') as f:
    #     pkl.dump(annotations, f)    
    if(base_args.Small):
        lower_area=0
        upper_area=32*32
        gt_x1=dfs_gt['x1']
        gt_x2=dfs_gt['x2']
        gt_y1=dfs_gt['y1']
        gt_y2=dfs_gt['y2']
        gt_areas=(gt_x2-gt_x1) *(gt_y2-gt_y1)
        print("Before removing medium and large bboxes: gt: "+ str(len(dfs_gt)))
        mask=(gt_areas>lower_area) & (gt_areas<upper_area)
        dfs_gt_small=dfs_gt[mask]
        print("After removing medium and large bboxes: gt: "+ str(len(dfs_gt_small)))
        
        
        pred_x1=dfs_pred['x1']
        pred_x2=dfs_pred['x2']
        pred_y1=dfs_pred['y1']
        pred_y2=dfs_pred['y2']
        print("Before removing medium and large bboxes: preds: "+  str(len(dfs_pred)))
        pred_areas=(pred_x2-pred_x1) *(pred_y2-pred_y1)
        mask=(pred_areas>lower_area) & (pred_areas<upper_area)
        dfs_pred_small=dfs_pred[mask]
        
        
        print("After removing medium and large bboxes: preds: "+ str(len(dfs_pred_small)))
        
        result, total_tp, total_fp, total_gt=compare_dfs(dfs_gt_small,dfs_pred_small) 
        with open(base_args.DetectEvalPthSmall, "w") as f:
            f.write("Camera ID                AP \n")
            f.write(str("Camera") + ": " +str(result)  +'\n')
            f.write("Average AP: " + str(np.mean(result))+ '\n')
            f.write("Total TP :" + str(total_tp)+'\n')
            f.write("Total_FP :"+ str(total_fp) + '\n')
            f.write("Total GT :" + str(total_gt  ))    
    if(base_args.Medium):
        lower_area=32*32
        upper_area=96*96
        gt_x1=dfs_gt['x1']
        gt_x2=dfs_gt['x2']
        gt_y1=dfs_gt['y1']
        gt_y2=dfs_gt['y2']
        gt_areas=(gt_x2-gt_x1) *(gt_y2-gt_y1)
        print("Before removing small and large bboxes: gt: "+ str(len(dfs_gt)))
        mask=(gt_areas>lower_area) & (gt_areas<upper_area)
        dfs_gt_small=dfs_gt[mask]
        print("After removing small and large bboxes: gt: "+ str(len(dfs_gt_small)))
        
        
        pred_x1=dfs_pred['x1']
        pred_x2=dfs_pred['x2']
        pred_y1=dfs_pred['y1']
        pred_y2=dfs_pred['y2']
        print("Before removing small and large bboxes: preds: "+  str(len(dfs_pred)))
        pred_areas=(pred_x2-pred_x1) *(pred_y2-pred_y1)
        mask=(pred_areas>lower_area) & (pred_areas<upper_area)
        dfs_pred_small=dfs_pred[mask]
        
        
        print("After removing small and large bboxes: preds: "+ str(len(dfs_pred_small)))
        
        result, total_tp, total_fp, total_gt=compare_dfs(dfs_gt_small,dfs_pred_small) 
        with open(base_args.DetectEvalPthMedium, "w") as f:
            f.write("Camera ID                AP \n")
            f.write(str("Camera") + ": " +str(result)  +'\n')
            f.write("Average AP: " + str(np.mean(result))+ '\n')
            f.write("Total TP :" + str(total_tp)+'\n')
            f.write("Total_FP :"+ str(total_fp) + '\n')
            f.write("Total GT :" + str(total_gt  ))    

    if(base_args.Large):
        lower_area=96*96
        upper_area=np.inf
        gt_x1=dfs_gt['x1']
        gt_x2=dfs_gt['x2']
        gt_y1=dfs_gt['y1']
        gt_y2=dfs_gt['y2']
        gt_areas=(gt_x2-gt_x1) *(gt_y2-gt_y1)
        print("Before removing small and medium bboxes: gt: "+ str(len(dfs_gt)))
        mask=(gt_areas>lower_area) & (gt_areas<upper_area)
        dfs_gt_small=dfs_gt[mask]
        print("After removing small and medium bboxes: gt: "+ str(len(dfs_gt_small)))
        
        
        pred_x1=dfs_pred['x1']
        pred_x2=dfs_pred['x2']
        pred_y1=dfs_pred['y1']
        pred_y2=dfs_pred['y2']
        print("Before removing small and medium bboxes: preds: "+  str(len(dfs_pred)))
        pred_areas=(pred_x2-pred_x1) *(pred_y2-pred_y1)
        mask=(pred_areas>lower_area) & (pred_areas<upper_area)
        dfs_pred_small=dfs_pred[mask]
        
        
        print("After removing small and medium bboxes: preds: "+ str(len(dfs_pred_small)))
        
        result, total_tp, total_fp, total_gt=compare_dfs(dfs_gt_small,dfs_pred_small) 
        with open(base_args.DetectEvalPthLarge, "w") as f:
            f.write("Camera ID                AP \n")
            f.write(str("Camera") + ": " +str(result)  +'\n')
            f.write("Average AP: " + str(np.mean(result))+ '\n')
            f.write("APs(50:95):" + str(result)+"\n")            
            f.write("Total TP :" + str(total_tp)+'\n')
            f.write("Total_FP :"+ str(total_fp) + '\n')
            f.write("Total GT :" + str(total_gt  ))    
    if(not base_args.Large and not base_args.Medium and not base_args.Small):
        result, total_tp, total_fp, total_gt, aps=compare_dfs(dfs_gt,dfs_pred)    
        print(base_args.DetectEvalPth)
        with open(base_args.DetectEvalPth, "w") as f:
            f.write("Camera ID                AP \n")
            f.write(str("Camera") + ": " +str(result)  +'\n')
            f.write("Average AP: " + str(np.mean(result))+ '\n')
            f.write("APs(50:95):" + str(aps)+"\n")
            f.write("Total TP :" + str(total_tp)+'\n')
            f.write("Total_FP :"+ str(total_fp) + '\n')
            f.write("Total GT :" + str(total_gt  ))    
        
    # if(args.EvalRois):
    #     print(args.EvalRoiPth)
    #     with open(args.EvalRoiPth, "w") as f:
    #         f.write("Camera ID                AP \n")
    #         f.write(str("Camera") + ": " +str(result)  +'\n')
    #         f.write("Average AP: " + str(np.mean(result))+ '\n')
    #         f.write("Total TP :" + str(total_tp)+'\n')
    #         f.write("Total_FP :"+ str(total_fp) + '\n')
    #         f.write("Total GT :" + str(total_gt  ))    
    # if(args.EvalIntersection):
    #     dfs_gt_intersection=[]
    #     pred_dfs_intersections=[]
    #     dfs_gt_not_intersection=[]
    #     pred_dfs_not_intersections=[]
    #     for gt in dfs_gt:
    #         gt1,gt2 =remove_out_of_ROI(gt, args.MetaData["roi"])
    #         dfs_gt_intersection.append(gt1)
    #         dfs_gt_not_intersection.append(gt2)
    #     for pred in dfs_pred:
    #         pred1,pred2=remove_out_of_ROI(pred, args.MetaData["roi"])
    #         pred_dfs_intersections.append(pred1)
    #         pred_dfs_not_intersections.append(pred2)
    #     result, total_tp, total_fp, total_gt= compare_dfs(dfs_gt_intersection, pred_dfs_intersections,args.EvalDetPthPlot)
    #     with open(args.EvalDetIntPth1, "w+") as f:
    #         print(args.EvalDetIntPth1)            
    #         f.write("Camera ID                AP \n")
    #         f.write(str("Camera") + ": " +str(result)  +'\n')
    #         f.write("Average AP: " + str(np.mean(result))+ '\n')
    #         f.write("Total TP :" + str(total_tp)+'\n')
    #         f.write("Total_FP :"+ str(total_fp) + '\n')
    #         f.write("Total GT :" + str(total_gt  ))
    #     if(args.EvalRois):
    #         print(args.EvalDetRoiIntPth1)
    #         with open(args.EvalDetRoiIntPth1, "w") as f:
    #             f.write("Camera ID                AP \n")
    #             f.write(str("Camera") + ": " +str(result)  +'\n')
    #             f.write("Average AP: " + str(np.mean(result))+ '\n')
    #             f.write("Total TP :" + str(total_tp)+'\n')
    #             f.write("Total_FP :"+ str(total_fp) + '\n')
    #             f.write("Total GT :" + str(total_gt  ))    
                
    #     result, total_tp, total_fp, total_gt= compare_dfs(dfs_gt_not_intersection, pred_dfs_not_intersections,args.EvalDetPthPlot)
    #     with open(args.EvalDetIntPth2, "w+") as f:
    #         print(args.EvalDetIntPth2)
    #         f.write("Camera ID                AP \n")
    #         f.write(str("Camera") + ": " +str(result)  +'\n')
    #         f.write("Average AP: " + str(np.mean(result))+ '\n')
    #         f.write("Total TP :" + str(total_tp)+'\n')
    #         f.write("Total_FP :"+ str(total_fp) + '\n')
    #         f.write("Total GT :" + str(total_gt  ))
    #     if(args.EvalRois):
    #         print(args.EvalDetRoiIntPth2)
    #         with open(args.EvalDetRoiIntPth2, "w") as f:
    #             f.write("Camera ID                AP \n")
    #             f.write(str("Camera") + ": " +str(result)  +'\n')
    #             f.write("Average AP: " + str(np.mean(result))+ '\n')
    #             f.write("Total TP :" + str(total_tp)+'\n')
    #             f.write("Total_FP :"+ str(total_fp) + '\n')
    #             f.write("Total GT :" + str(total_gt  ))    
        
    return SucLog("Detection Evaluation Successful")


# From https://github.com/rafaelpadilla/Object-Detection-Metrics, take the recall and precision and first smooth and then calculate area under smoothed curve. 
def CalculateAveragePrecision(rec, prec):
    mrec = []
    mrec.append(0)
    [mrec.append(e) for e in rec]
    mrec.append(1)
    mpre = []
    mpre.append(0)
    [mpre.append(e) for e in prec]
    mpre.append(0)
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = []
    for i in range(len(mrec) - 1):
        if mrec[1+i] != mrec[i]:
            ii.append(i + 1)
    ap = 0
    for i in ii:
        ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
    # return [ap, mpre[1:len(mpre)-1], mrec[1:len(mpre)-1], ii]
    return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]
def compare_difs(dfs_np, pred_np):
    thresholds= np.asarray([ x/100.0 for x in range(50,105,5)])
    print(dfs_np,pred_np)    
def compare_dfs(dfs_gts, pred_dfs):
    thresholds= np.asarray([ x/100.0 for x in range(50,100,5)])
    # classes=[ 0,1, 2,3, 5, 7]
    # classes=np.unique(pred_dfs['class'])
    
    classes=np.unique(dfs_gts['class'])
    # print(gt_classes)
    # z=0
    # thresholds= np.asarray([0.5])
    class_counts=[]
    class_aps=[]
    total_tp=0
    total_fp=0
    total_gt=0
    class_aps=[]
    class_counts=[]
    aps_threshold=[]
    print(thresholds)
    for c in classes:
        print(c)
        aps=[]    
        gt=dfs_gts[dfs_gts['class']==c]
        pred=pred_dfs[pred_dfs['class']==c]
        gt.insert(0, 'index', range(0, len(gt)))
        # gt_byframe=dict(tuple(gt.groupby('fn')))
        pred_byframe=dict(tuple(pred.groupby('fn')))
        # gt_frames=gt_byframe.keys()
        pred_frames=pred_byframe.keys()
        # common_frames=list(set(gt_frames).intersection(set(pred_frames)))
        n_gt=len(gt)
        n_pred=len(pred)
        all_predictions=[]
        # fn_frames=list(set(gt_frames)- set(pred_frames))
        # fp_frames=list(set(pred_frames)- set(gt_frames))
        pred_bboxes=sorted(np.asarray(pred), key=lambda x:x[2], reverse=True)
        TP= np.zeros((len(pred_bboxes), len(thresholds)))
        FP= np.zeros((len(pred_bboxes), len(thresholds)))
        # print(pred)
        # print(gt)
        g_full=np.asarray(gt)

        used_ids=np.zeros((len(gt), len(thresholds)) , dtype=bool)
        for i in range(len(pred_bboxes)):
            pred_bbox=pred_bboxes[i]
            g_= gt[gt['df_id']==pred_bbox[-1]]
            gts=np.asarray(g_[(g_['fn']==pred_bbox[0])])
            
            iou_max=0
            p=pred_bbox[3:-1]
            iou_idx=-1            
            for k, gt_bbox in enumerate(gts):
                g=gt_bbox[[0,4,5,6,7]]    
                # print(g)
                iou=calculate_iou(p,g[1:])
                if iou>iou_max:
                    iou_max=iou
                    iou_idx=int(gt_bbox[0])
            iou_eval=iou_max>thresholds
            if(iou_idx!=-1):
                TP[i]= ~used_ids[iou_idx] &  iou_eval
                FP[i]= 1-(TP[i])
                used_ids[iou_idx]= used_ids[iou_idx] | iou_eval                    
            else:
                FP[i]=np.ones((len(thresholds)))
                TP[i]=1-FP[i]
        tp_s= np.sum(TP, axis=0)
        fp_s= np.sum(FP, axis=0)
        tp_cs=np.cumsum(TP,axis=0)
        fp_cs=np.cumsum(FP,axis=0)     
        total_tp+=tp_s
        total_fp+=fp_s
        total_gt+=n_gt
        recall=tp_cs/n_gt
        precision=np.divide(tp_cs,(tp_cs+fp_cs))
        recall_thresholds = np.linspace(0.0,
                                        1.00,
                                        int(np.round((1.00 - 0.0) / 0.01)) + 1,
                                        endpoint=True)            
    
        for rec, prec in zip(recall.T,precision.T):
            i_pr = np.maximum.accumulate(prec[::-1])[::-1]   
            rec_idx = np.searchsorted(rec, recall_thresholds, side="left")      
            n_recalls = len(recall_thresholds)             
            i_pr = np.array([i_pr[r] if r < len(i_pr) else 0 for r in rec_idx])  
            [ap, mpre, mrec, ii]=CalculateAveragePrecision(rec,prec)
            print(np.mean(i_pr), ap)
            aps.append(np.mean(i_pr))
        if n_gt>0:
            class_aps.append(np.mean(aps))
            aps_threshold.append(aps)
            print("class ", c, " n_gt ", n_gt, "n_pred ", n_pred, ' ap ', np.mean(aps))
        class_counts.append(n_pred)
    mAP=np.average(class_aps)
    aps_threshold=np.average(np.array(aps_threshold),axis=0)
    print(total_tp, total_fp, total_gt)
    print(class_aps)
    print(aps_threshold)
    return mAP, total_tp, total_fp, total_gt, aps_threshold
            # if(pred_bbox[0] in fp_frames):
            #     FP[i]=np.ones(len(thresholds))
            # else:
        #     frame_gt=np.asarray([pred_bbox[0]])
        #     iou_max=0
        #     p=pred_bbox[3:]
        #     iou_idx=-1
        #     for k,gt_bbox in enumerate(frame_gt):
        #         # g=gt_bbox[1:7]
        #         g=gt_bbox[[0,4,5,6,7]]    
        #         # print(g)
        #         # input()  
        #         iou=calculate_iou(p,g[1:])
        #         if iou>iou_max:
        #             iou_max=iou
        #             iou_idx=int(gt_bbox[0])
        #     iou_eval=iou_max>thresholds
            
        #     if(iou_idx!=-1):
        #         TP[i]= ~used_ids[iou_idx] &  iou_eval
        #         FP[i]= 1-(TP[i])
        #         used_ids[iou_idx]= used_ids[iou_idx] | iou_eval                    
        #     else:
        #         FP[i]=np.ones((len(thresholds)))
        #         TP[i]=1-FP[i]
        # tp_s= np.sum(TP, axis=0)
        # fp_s= np.sum(FP, axis=0)
        
        # tp_cs=np.cumsum(TP,axis=0)
        # fp_cs=np.cumsum(FP,axis=0)     
        # total_tp+=tp_s
        # total_fp+=fp_s
        # total_gt+=n_gt
        # recall=tp_cs/n_gt
        # precision=np.divide(tp_cs,(tp_cs+fp_cs))
        # aps=[]
        # count=0
        
        # for rec, prec in zip(recall.T,precision.T):
        #     [ap, mpre, mrec, ii]=CalculateAveragePrecision(rec,prec)
        #     aps.append(ap)
        #     if(count==5 and c==2):
        #         fig, ax = plt.subplots()
        #         ax.plot(rec, prec, color='purple')                    
        #     count=count+1
        # total_ap=np.mean(aps)
        
        # if not np.isnan(total_ap):
        #     class_aps.append(total_ap)
        #     class_counts.append(len(gt))
        # else:
        #     class_aps.append(0)
        #     class_counts.append(len(gt))
    
    # camera_aps=[]
    # classes=[ 0,1, 2,3, 5, 7]
    # total_gt=np.zeros((len(thresholds)))
    # total_tp=np.zeros((len(thresholds)))
    # total_fp=np.zeros((len(thresholds)))
    # for gtfull, predfull in zip(dfs_gts, pred_dfs):
    #     class_counts=[]
    #     class_aps=[]
    #     for c in classes:
    #         gt= gtfull[gtfull['class']==c]

    #         pred= predfull[predfull['class']==c]
    #         z=z+1
    #         gt.insert(0, 'index', range(0, 0 + len(gt)))            
    #         gt_byframe=dict(tuple(gt.groupby('fn')))
    #         pred_byframe=dict(tuple(pred.groupby('fn')))
    #         gt_frames=gt_byframe.keys()
    #         pred_frames=pred_byframe.keys()
    #         common_frames=list(set(gt_frames).intersection(set(pred_frames)))
    #         n_gt=len(gt)
    #         all_predictions=[]
    #         fn_frames=list(set(gt_frames)- set(pred_frames))
    #         fp_frames=list(set(pred_frames)- set(gt_frames))

    #         pred_bboxes=sorted(np.asarray(pred), key=lambda x:x[2], reverse=True)
    #         TP= np.zeros((len(pred_bboxes), len(thresholds)))
    #         FP= np.zeros((len(pred_bboxes), len(thresholds)))
    #         # print(pred)
    #         # print(gt)
    #         used_ids=np.zeros((len(gt), len(thresholds)) , dtype=bool)
    #         for i in range(len(pred_bboxes)):
    #             pred_bbox=pred_bboxes[i]
    #             if(pred_bbox[0] in fp_frames):
    #                 FP[i]=np.ones(len(thresholds))
    #             else:
    #                 frame_gt=np.asarray(gt_byframe[pred_bbox[0]])
    #                 iou_max=0
    #                 p=pred_bbox[3:]
    #                 iou_idx=-1
    #                 for k,gt_bbox in enumerate(frame_gt):
    #                     # g=gt_bbox[1:7]
    #                     g=gt_bbox[[0,4,5,6,7]]    
    #                     # print(g)
    #                     # input()  
    #                     iou=calculate_iou(p,g[1:])
    #                     if iou>iou_max:
    #                         iou_max=iou
    #                         iou_idx=int(gt_bbox[0])
    #                 iou_eval=iou_max>thresholds
                    
    #                 if(iou_idx!=-1):
    #                     TP[i]= ~used_ids[iou_idx] &  iou_eval
    #                     FP[i]= 1-(TP[i])
    #                     used_ids[iou_idx]= used_ids[iou_idx] | iou_eval                    
    #                 else:
    #                     FP[i]=np.ones((len(thresholds)))
    #                     TP[i]=1-FP[i]
    #         tp_s= np.sum(TP, axis=0)
    #         fp_s= np.sum(FP, axis=0)
            
    #         tp_cs=np.cumsum(TP,axis=0)
    #         fp_cs=np.cumsum(FP,axis=0)     
    #         total_tp+=tp_s
    #         total_fp+=fp_s
    #         total_gt+=n_gt
    #         recall=tp_cs/n_gt
    #         precision=np.divide(tp_cs,(tp_cs+fp_cs))
    #         aps=[]
    #         count=0
            
    #         for rec, prec in zip(recall.T,precision.T):
    #             [ap, mpre, mrec, ii]=CalculateAveragePrecision(rec,prec)
    #             aps.append(ap)
    #             if(count==5 and c==2):
    #                 fig, ax = plt.subplots()
    #                 ax.plot(rec, prec, color='purple')                    
    #             count=count+1
    #         total_ap=np.mean(aps)
            
    #         if not np.isnan(total_ap):
    #             class_aps.append(total_ap)
    #             class_counts.append(len(gt))
    #         else:
    #             class_aps.append(0)
    #             class_counts.append(len(gt))
    #     print(class_aps)
    #     camera_aps.append(np.average(class_aps, weights=class_counts))
    # return camera_aps, total_tp, total_fp, total_gt


def calculate_iou(pred,gt):
    assert len(pred)==4
    assert len(gt)==4
    inter_left=max(gt[0], pred[0])
    inter_right=min(gt[2], pred[2])
    inter_top= max(gt[1], pred[1])
    inter_bottom= min(gt[3], pred[3])    
    inter_area= (inter_right-inter_left +1) *(inter_bottom- inter_top +1) 
    gt_area= (gt[2]-gt[0] +1) * (gt[3]- gt[1] +1)
    pred_area=(pred[2]-pred[0] +1) * (pred[3]-pred[1]+1)
    res= inter_area/float(gt_area+pred_area - inter_area) if inter_left< inter_right and inter_top < inter_bottom else 0
    return res

def remove_out_of_ROI(df, roi):
    poly_path = mplPath.Path(np.array(roi))
    mask = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="rm oROI bbox"):
        x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
        p = [(x2+x1)/2, y2]
        if poly_path.contains_point(p):
            mask.append(True)
        else: mask.append(False)
    m=list(~np.array(mask))
    return df[mask], df[m]  

