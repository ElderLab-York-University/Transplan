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
def compare_dfs(dfs_gts, pred_dfs, plot_path):
    z=0
    f_scores=[]
    thresholds= np.asarray([ x/100.0 for x in range(50,105,5)])
    camera_aps=[]
    classes=[ 0,1, 2,3, 5, 7]
    total_gt=np.zeros((len(thresholds)))
    total_tp=np.zeros((len(thresholds)))
    total_fp=np.zeros((len(thresholds)))
    for gtfull, predfull in zip(dfs_gts, pred_dfs):
        class_counts=[]
        class_aps=[]
        for c in classes:
            gt= gtfull[gtfull['class']==c]

            pred= predfull[predfull['class']==c]
            z=z+1
            gt.insert(0, 'index', range(0, 0 + len(gt)))            
            gt_byframe=dict(tuple(gt.groupby('fn')))
            pred_byframe=dict(tuple(pred.groupby('fn')))
            gt_frames=gt_byframe.keys()
            pred_frames=pred_byframe.keys()
            common_frames=list(set(gt_frames).intersection(set(pred_frames)))
            n_gt=len(gt)
            all_predictions=[]
            fn_frames=list(set(gt_frames)- set(pred_frames))
            fp_frames=list(set(pred_frames)- set(gt_frames))

            pred_bboxes=sorted(np.asarray(pred), key=lambda x:x[2], reverse=True)
            TP= np.zeros((len(pred_bboxes), len(thresholds)))
            FP= np.zeros((len(pred_bboxes), len(thresholds)))
            # print(pred)
            # print(gt)
            used_ids=np.zeros((len(gt), len(thresholds)) , dtype=bool)
            for i in range(len(pred_bboxes)):
                pred_bbox=pred_bboxes[i]
                if(pred_bbox[0] in fp_frames):
                    FP[i]=np.ones(len(thresholds))
                else:
                    frame_gt=np.asarray(gt_byframe[pred_bbox[0]])
                    iou_max=0
                    p=pred_bbox[3:]
                    iou_idx=-1
                    for k,gt_bbox in enumerate(frame_gt):
                        # g=gt_bbox[1:7]
                        g=gt_bbox[[0,4,5,6,7]]    
                        # print(g)
                        # input()  
                        iou=calculate_iou(p,g[1:])
                        if iou>iou_max:
                            iou_max=iou
                            iou_idx=int(gt_bbox[0])
                            # if(iou_idx==269):
                            #     print(g)
                            #     print(frame_gt)
                            #     input()
                    iou_eval=iou_max>thresholds
                    
                    if(iou_idx!=-1):
                        TP[i]= ~used_ids[iou_idx] &  iou_eval
                        FP[i]= 1-(TP[i])
                        used_ids[iou_idx]= used_ids[iou_idx] | iou_eval                    
                    else:
                        FP[i]=np.ones((len(thresholds)))
                        TP[i]=1-FP[i]
                    # print("__________________________________________________________________________")
                    # print(TP[i])
                    # print(FP[i])
                    # print("__________________________________________________________________________")
                    # print(iou_idx)
                    # print(used_ids[iou_idx])
                    # print(used_ids[iou_idx])
                    # input()
                    # print("__________________________________________________________________________")

                    # TP[i]=iou_eval
                    # FP[i]= np.invert(iou_eval)
            tp_s= np.sum(TP, axis=0)
            fp_s= np.sum(FP, axis=0)
            
            tp_cs=np.cumsum(TP,axis=0)
            fp_cs=np.cumsum(FP,axis=0)     
            total_tp+=tp_s
            total_fp+=fp_s
            total_gt+=n_gt
            # recall= tp_s/n_gt
            # precision=tp_s/(tp_s+fp_s)
            recall=tp_cs/n_gt
            precision=np.divide(tp_cs,(tp_cs+fp_cs))
            aps=[]
            count=0
            
            for rec, prec in zip(recall.T,precision.T):
                [ap, mpre, mrec, ii]=CalculateAveragePrecision(rec,prec)
                aps.append(ap)
                if(count==5 and c==2):
                    fig, ax = plt.subplots()
                    ax.plot(rec, prec, color='purple')

                    # add axis labels to plot
                    ax.set_title('Precision-Recall Curve Class: Cars')
                    ax.set_ylabel('Precision')
                    ax.set_xlabel('Recall')
                    fig.savefig(plot_path)
                    
                count=count+1
            total_ap=np.mean(aps)
            if(n_gt>0):
                class_aps.append(total_ap)
            # if not np.isnan(total_ap):
            #     class_aps.append(total_ap)
            #     class_counts.append(len(gt))
            # else:
            #     class_aps.append(0)
            #     class_counts.append(len(gt))
        print(class_aps)
        camera_aps.append(np.mean(class_aps))
        # print("Recalls for ", s_threshold, " camera number " , z, recall)
        # print("Precisions for ", s_threshold, " camera number " , z, precision)
        # f_betas=((1+beta**2)* (recall*precision))/(beta**2*(recall+precision))
        # f_beta= np.mean(f_betas)
        # f_scores.append(f_beta)
    # total_recall=total_tp/total_gt
    # total_precision=total_tp/(total_fp+total_fp)

    # total_f_betas=((1*beta**2)*(total_recall*total_precision)/(beta**2*(total_recall+total_precision)))
    # total_f_beta=np.mean(total_f_betas)
    return camera_aps, total_tp, total_fp, total_gt
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

def evaluate_detections(args):
    args_gt = get_args_gt(args)
    dfs_pred =[]
    dfs_gt   =[]
    cam_ids  =[]
    
    # get dfs and cam_ids for gt
    # get dfs and cam_ids for prediction
    dfs_pred =[]
    dfs_gt   =[]
    cam_ids  =[]
    print(args_gt.DetectionPkl)
        
    if(args.MaskGT):
        df=pd.read_pickle(args_gt.MaskedGT)
    else:
        df = pd.read_pickle(args_gt.DetectionPkl)
    vid = cv2.VideoCapture(args_gt.Video)
    frame_bottom = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_right = vid.get(cv2.CAP_PROP_FRAME_WIDTH) 
    frame_left=0
    frame_top=0
    if(args.BboxMin is not None):
        # print([((df['x2']-df['x1']) * df['y2']-df['y1'])])
        a= args.BboxMax if args.BboxMax is not None else max((df['x2']-df['x1'])*(df['y2']-df['y1']))
        df=(df[(((df['x2']-df['x1']) * df['y2']-df['y1']) < a) & (((df['x2']-df['x1']) * df['y2']-df['y1']) > args.BboxMin) ])
        
        # df=df[(((df['x2']-df['x1'])*(df['y2']- df['y1'])> args.BboxMin) & (df['x2']-df['x1'])*(df['y2']- df['y1'])> args.BboxMin)]
    print(len(df))
    # df=df[(df['x1']>frame_left) & (df['y1']>frame_top) & (df['x2'] < frame_right) & (df['y2'] < frame_bottom)]
    # df=df[df['fn']==0]
    print(df)
    print(len(df))               
    # print(df)
    # df2=pd.read_pickle(arg.TrackingPkl)
    
    # print(df2)EvalPth
    # input()
    # df=df[df['fn']<=2]
    dfs_gt.append(df)
            
    if(args.EvalRois):
        df=pd.read_pickle(args.DetectionPklRois)
        
    else:
        df = pd.read_pickle(args.DetectionPkl)
    print(df)
        
    g=(dfs_gt[0])
    max_fn=g.max(axis=0)['fn']
    min_fn=g.min(axis=0)['fn']
    df=df[(df['fn'] >= min_fn) & (df['fn'] <=max_fn ) &((df['fn']-min_fn)%6==0)]
    # if(args.classes_to_eval is not None and len(args.classes_to_eval)>0):
    #     df=df[df['class'].isin(args.classes_to_eval)]
    if(args.DetTh is not None):
        df=df[df['score']>args.DetTh]
        # print(len(df))
        # print(len(df[df['score']>0.5]))        
        # print(g)
        # df=df[df['fn']<=2]
    if(args.BboxMin is not None):
        # print([((df['x2']-df['x1']) * df['y2']-df['y1'])])
        a= args.BboxMax if args.BboxMax is not None else max((df['x2']-df['x1'])*(df['y2']-df['y1']))
        df=(df[(((df['x2']-df['x1']) * df['y2']-df['y1']) < a) & (((df['x2']-df['x1']) * df['y2']-df['y1']) > args.BboxMin) ])
        
    df=df[(df['x1']>frame_left) & (df['y1']>frame_top) & (df['x2'] < frame_right) & (df['y2'] < frame_bottom)]        
    # df=df[df['fn']==0]
    print(df)        
    dfs_pred.append(df)
    

    # dfs_gt= prepare_gt_for_evaluation(dfs_gt)
    pred_dfs= dfs_pred
    print(args.EvalDetPthPlot)
    result, total_tp, total_fp, total_gt= compare_dfs(dfs_gt, pred_dfs,args.EvalDetPthPlot)
    with open(args.EvalDetPth, "w") as f:
        f.write("Camera ID                AP \n")
        f.write(str("Camera") + ": " +str(result)  +'\n')
        f.write("Average AP: " + str(np.mean(result))+ '\n')
        f.write("Total TP :" + str(total_tp)+'\n')
        f.write("Total_FP :"+ str(total_fp) + '\n')
        f.write("Total GT :" + str(total_gt  ))    
    if(args.EvalRois):
        print(args.EvalRoiPth)
        with open(args.EvalRoiPth, "w") as f:
            f.write("Camera ID                AP \n")
            f.write(str("Camera") + ": " +str(result)  +'\n')
            f.write("Average AP: " + str(np.mean(result))+ '\n')
            f.write("Total TP :" + str(total_tp)+'\n')
            f.write("Total_FP :"+ str(total_fp) + '\n')
            f.write("Total GT :" + str(total_gt  ))    
    if(args.EvalIntersection):
        dfs_gt_intersection=[]
        pred_dfs_intersections=[]
        dfs_gt_not_intersection=[]
        pred_dfs_not_intersections=[]
        for gt in dfs_gt:
            gt1,gt2 =remove_out_of_ROI(gt, args.MetaData["roi"])
            dfs_gt_intersection.append(gt1)
            dfs_gt_not_intersection.append(gt2)
        for pred in dfs_pred:
            pred1,pred2=remove_out_of_ROI(pred, args.MetaData["roi"])
            pred_dfs_intersections.append(pred1)
            pred_dfs_not_intersections.append(pred2)
        result, total_tp, total_fp, total_gt= compare_dfs(dfs_gt_intersection, pred_dfs_intersections,args.EvalDetPthPlot)
        with open(args.EvalDetIntPth1, "w+") as f:
            print(args.EvalDetIntPth1)            
            f.write("Camera ID                AP \n")
            f.write(str("Camera") + ": " +str(result)  +'\n')
            f.write("Average AP: " + str(np.mean(result))+ '\n')
            f.write("Total TP :" + str(total_tp)+'\n')
            f.write("Total_FP :"+ str(total_fp) + '\n')
            f.write("Total GT :" + str(total_gt  ))
        if(args.EvalRois):
            print(args.EvalDetRoiIntPth1)
            with open(args.EvalDetRoiIntPth1, "w") as f:
                f.write("Camera ID                AP \n")
                f.write(str("Camera") + ": " +str(result)  +'\n')
                f.write("Average AP: " + str(np.mean(result))+ '\n')
                f.write("Total TP :" + str(total_tp)+'\n')
                f.write("Total_FP :"+ str(total_fp) + '\n')
                f.write("Total GT :" + str(total_gt  ))    
                
        result, total_tp, total_fp, total_gt= compare_dfs(dfs_gt_not_intersection, pred_dfs_not_intersections,args.EvalDetPthPlot)
        with open(args.EvalDetIntPth2, "w+") as f:
            print(args.EvalDetIntPth2)
            f.write("Camera ID                AP \n")
            f.write(str("Camera") + ": " +str(result)  +'\n')
            f.write("Average AP: " + str(np.mean(result))+ '\n')
            f.write("Total TP :" + str(total_tp)+'\n')
            f.write("Total_FP :"+ str(total_fp) + '\n')
            f.write("Total GT :" + str(total_gt  ))
        if(args.EvalRois):
            print(args.EvalDetRoiIntPth2)
            with open(args.EvalDetRoiIntPth2, "w") as f:
                f.write("Camera ID                AP \n")
                f.write(str("Camera") + ": " +str(result)  +'\n')
                f.write("Average AP: " + str(np.mean(result))+ '\n')
                f.write("Total TP :" + str(total_tp)+'\n')
                f.write("Total_FP :"+ str(total_fp) + '\n')
                f.write("Total GT :" + str(total_gt  ))    
                

        


        
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
        
