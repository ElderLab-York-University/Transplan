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
    x1, y1, x2, y2, x, y, id, fn
    '''
    labels = defaultdict(list)
    for df, cam_id in zip(dfs, cam_ids):
        raw_list = []
        for i, row in df.iterrows():
            raw_list.append({'FrameId':row.fn, 'Id':int(row.id), 'X':int(float(row.x1)), 'Y':int(float(row.y1)), 'Width':int(float(row.x2-row.x1)), 'Height':int(float(row.y2-row.y1)), 'Confidence':1.0})
        labels[cam_id].extend(raw_list)
    return OrderedDict([(cam_id, pd.DataFrame(rows).set_index(['FrameId', 'Id'])) for cam_id, rows in labels.items()])


def evaluate(args):
    # get args_mc_gt 
    args_gt = get_args_gt(args)
    # get dfs and cam_ids for gt
    # get dfs and cam_ids for prediction
    dfs_pred =[]
    dfs_gt   =[]
    cam_ids  =[]
    
    # for arg in args_mc_gt:
    df = pd.read_pickle(args_gt.TrackingPkl)
    dfs_gt.append(df)
    
    # for a,arg in enumerate(args_mc):
    df = pd.read_pickle(args_gt.TrackingPkl)
    # print(df)
    
    # print(df)
    df=pd.read_pickle(args.TrackingPkl)
    g=(dfs_gt[0])
    max_fn=g.max(axis=0)['fn']
    min_fn=g.min(axis=0)['fn']
    df=df[(df['fn'] >= min_fn) & (df['fn'] <=max_fn ) &((df['fn']-min_fn)%6==0)]
    cam_ids.append(1)
    # dfs_pred.append(df)
    # cam_ids.append(arg.CamID)
        # print(df)
        # df = pd.read_pickle(arg.TrackingPkl)
    dfs_pred.append(df)
        # cam_ids.append(arg.CamID)
    
    # print(df)
    # prepare dfs for mot metrics(transfer from local format to mot format)
    gt_dfs = prepare_df_for_motmetric(dfs_gt, cam_ids)
    pred_dfs = prepare_df_for_motmetric(dfs_pred, cam_ids)

    accs, names = compare_dataframes(gt_dfs, pred_dfs)
    mh = mm.metrics.create()
    summary = mh.compute_many(accs, names=names, metrics=mm.metrics.motchallenge_metrics, generate_overall=True)

    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)


    dfs_pred =[]
    dfs_gt   =[]
    cam_ids  =[]
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



    with open(args.EvalPth, "w") as f:
        f.write(strsummary)
        # f.write("\n")
        # f.write(strsummary_mc)
        
    