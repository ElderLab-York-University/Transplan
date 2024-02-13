from Utils import *
from Libs import *
import Track
import Maps
import DSM
import Segment
import Detect
# import homographygui.tabbed_ui_func as tui

#hints the vars set for homography are 
    # args.HomographyStreetView
    # args.HomographyTopView
    # args.HomographyTXT
    # args.HomographyNPY
    # args.HomographyCSV
    # args.ReprojectedPoints
    # args.VisHomographyPth
    # args.ReprojectedPkl

def homographygui(args):
    # assume homography repo is made in results
    # check if homography pair pictures are available with the video
    if not os.path.exists(args.HomographyTopView):
        print(ProcLog("intersection topview is not given; will fetch from gmaps"))
        download_top_view_image(args)
        
    if not os.path.exists(args.HomographyStreetView):
        print(ProcLog("intersection streetview is not given; will choose videos first frame"))
        save_frame_from_video(args)
        

    # lunch homography gui
    lunch_homographygui(args)
    return SucLog("Homography GUI executed successfully")    
    # if all good lunch homographGUI

def download_top_view_image(args):
    center = args.MetaData['center']
    file_name = args.HomographyTopView
    print(f"file_name of donwloaded top view : {file_name}")
    if args.TopView == "GoogleMap":
        Maps.download_image(center=center, file_name=file_name)
    else:
        print("automatic download only supported for GoogleMap topview")

def lunch_homographygui(args):
    street = os.path.abspath(args.HomographyStreetView)
    top = os.path.abspath(args.HomographyTopView)
    txt = os.path.abspath(args.HomographyTXT)
    npy = os.path.abspath(args.HomographyNPY)
    csv = os.path.abspath(args.HomographyCSV)
    cwd = os.getcwd()
    os.chdir("./homographygui/")
    os.system(f"python3 main.py --StreetView='{street}' --TopView='{top}' --Txt='{txt}' --Npy='{npy}' --Csv='{csv}'")
    os.chdir(cwd)

def iou(det_row, mask_row):
    box1 = det_row
    box2 = mask_row
    # Calculate area of each bounding box
    area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
    area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)

    # Calculate intersection of the two bounding boxes
    intersection_x1 = np.maximum(box1.x1, box2.x1)
    intersection_y1 = np.maximum(box1.y1, box2.y1)
    intersection_x2 = np.minimum(box1.x2, box2.x2)
    intersection_y2 = np.minimum(box1.y2, box2.y2)
    intersection_w = np.maximum(0, intersection_x2 - intersection_x1)
    intersection_h = np.maximum(0, intersection_y2 - intersection_y1)
    intersection_area = intersection_w * intersection_h

    # Calculate union of the two bounding boxes
    union_area = area1 + area2 - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou
    

def get_mask_with_max_iou(det_row, mask_df, th=0.5):
    ious = []
    for i, mask_row in mask_df.iterrows():
        ious.append(iou(det_row, mask_row))
    if len(ious) ==0: # no segmentations in mask_df
        return None

    max_idx = np.array(ious).argmax()
    if ious[max_idx] < th: # if the best match is not yet good enough
        return None

    return mask_df.iloc[max_idx]

def get_det_with_max_iou(det_row, detections_df, th=0.5):
    # we want to find a match to det_row from detections_df
    ious = []
    for i, mask_row in detections_df.iterrows():
        ious.append(iou(det_row, mask_row))
    if len(ious) ==0: # no segmentations in mask_df
        return None

    max_idx = np.array(ious).argmax()
    if ious[max_idx] < th: # if the best match is not yet good enough
        return None

    return detections_df.iloc[max_idx]

def find_cp_BottomSeg(det_row, args):
    fn = int(det_row["fn"])
    # get the right segmentation file
    seg_fn_path = Segment.get_results_path_with_frame(args.SegmentPkl, fn)
    mask_df = pd.read_pickle(seg_fn_path)
    mask_row = get_mask_with_max_iou(det_row, mask_df)
    if mask_row is not None: # found a matching mask
        det_x_mid = int((det_row.x1 + det_row.x2)/2)
        if det_x_mid >= mask_row.x1 and det_x_mid <= mask_row.x2:
            mask_x_idx_to_look = int(det_x_mid - mask_row.x1)
            mask_y_min_idx_to_look = int(min(max(det_row.y1 - mask_row.y1, 0), mask_row.y2 - mask_row.y1))
            mask_y_max_idx_to_look = int(min(max(det_row.y2 - mask_row.y1, 0), mask_row.y2 - mask_row.y1))
            mask_line_to_look = mask_row["mask"][mask_y_min_idx_to_look:mask_y_max_idx_to_look, mask_x_idx_to_look]
            indices = np.where(mask_line_to_look == 1)
            if indices[0].size > 0:
                # Get the largest index
                largest_index = np.max(indices)
                y = largest_index + mask_y_min_idx_to_look + mask_row.y1
                x = mask_row.x1 + mask_x_idx_to_look
                return (x, y)
            
    # if could not find a contact point on Seg mask, return bottom point        
    x, y = (det_row['x2']+det_row['x1'])/2, det_row['y2']
    return (x, y)

def find_cp_LineSeg(det_row, args):
    fn = int(det_row["fn"])
    # get the right segmentation file
    seg_fn_path = Segment.get_results_path_with_frame(args.SegmentPkl, fn)
    mask_df = pd.read_pickle(seg_fn_path)
    mask_row = get_mask_with_max_iou(det_row, mask_df)
    if mask_row is not None: # found a matching mask
            mask_y_min_idx_to_look = int(min(max(det_row.y1 - mask_row.y1, 0), mask_row.y2 - mask_row.y1))
            mask_y_max_idx_to_look = int(min(max(det_row.y2 - mask_row.y1, 0), mask_row.y2 - mask_row.y1))
            mask_x_min_idx_to_look = int(min(max(det_row.x1 - mask_row.x1, 0), mask_row.x2 - mask_row.x1))
            mask_x_max_idx_to_look = int(min(max(det_row.x2 - mask_row.x1, 0), mask_row.x2 - mask_row.x1))

            mask_part_to_look = mask_row["mask"][mask_y_min_idx_to_look:mask_y_max_idx_to_look, mask_x_min_idx_to_look:mask_x_max_idx_to_look]
            # transpose mask so that the indices are sorted on x coordinate
            indices_x, indices_y = np.where(mask_part_to_look.T == 1)
            if len(indices_x) > 0:
                min_indices_x, max_indices_x = min(indices_x), max(indices_x)
                min_indices_x_max_y = indices_y[np.where(indices_x == min_indices_x)[0][-1]]
                max_indices_x_max_y = indices_y[np.where(indices_x == max_indices_x)[0][-1]]

                x1 = min_indices_x + mask_x_min_idx_to_look + mask_row.x1
                y1 = min_indices_x_max_y + mask_y_min_idx_to_look + mask_row.y1

                x2 = max_indices_x + mask_x_min_idx_to_look + mask_row.x1
                y2 = max_indices_x_max_y + mask_y_min_idx_to_look + mask_row.y1

                x_cp, y_cp = int((x1 + x2)/2) , int((y1 + y2)/2)
                return (x_cp, y_cp)
            
    # if could not find a contact point on Seg mask, return bottom point        
    x, y = (det_row['x2']+det_row['x1'])/2, det_row['y2']
    return (x, y)

def get_contact_point(row, args):
    if args.ContactPoint == "BottomPoint":
        x, y = (row['x2']+row['x1'])/2, row['y2']
        return (x, y)
    
    elif args.ContactPoint == "BottomPoint3D":
        # row will have x1-8 and y1-8
        x = (row.x1 + row.x2 + row.x3 + row.x4)/4
        y = (row.y1 + row.y2 + row.y3 + row.y4)/4
        return (x, y)
    
    elif args.ContactPoint == "Center":
        x, y = (row['x2']+row['x1'])/2, (row['y2']+row['y1'])/2
        return (x, y)
    
    elif args.ContactPoint == "BottomSeg":
        x, y = find_cp_BottomSeg(row, args)
        return (x, y)
    
    elif args.ContactPoint == "LineSeg":
        x, y = find_cp_LineSeg(row, args)
        return (x, y)
    
    else: raise NotImplemented

def reproj_point(args, x, y, method, **kwargs):
    if method == "Homography":
        return reproj_point_homography(x, y, kwargs["M"])
        
    elif method == "DSM":
        return reproj_point_dsm(x, y, kwargs["GroundRaster"], kwargs["TifObj"])
    
    else: raise NotImplemented

def project_point(args, x, y, method, **kwargs):
    if method == "Homography":
        return project_point_homography(x, y, kwargs["M"])
        
    elif method == "DSM":
        raise NotImplementedError
        return project_point_dsm(x, y, None, None)
    
    else: raise NotImplemented

def reproj_point_homography(x, y, M):
    point = np.array([x, y, 1])
    new_point = M.dot(point)
    new_point /= new_point[2]
    return new_point

def project_point_homography(x, y, M):
    point = np.array([x, y, 1])
    Minv = np.linalg.inv(M)
    new_point  = Minv.dot(point)
    new_point /= new_point[2]
    return new_point

def reproj_point_dsm(x, y, img_ground_raster, orthophoto_win_tif_obj):
    u_int, v_int = int(x), int(y)
    # mathced_coord are the real-world coordinates we need to change them to top-view pixel coordinates(for consistancy reasons)
    matched_coord =  img_ground_raster[v_int, u_int]
    orthophoto_proj_idx = orthophoto_win_tif_obj.index(*matched_coord[:-1])
    # pass (orthophoto_proj_idx[1], orthophoto_proj_idx[0]) to csv.drawMarker
    return (orthophoto_proj_idx[1], orthophoto_proj_idx[0])

def project_point_dsm(x, y, img_ground_raster, orthophoto_win_tif_obj):
    raise NotImplementedError

def reproject_df(args, df, out_path, method):
    # we will load M and Raster here and pass it on to speed up the projection
    M = None
    GroundRaster = None
    orthophoto_win_tif_obj = None

    if method == "Homography":
        homography_path = args.HomographyNPY
        M = np.load(homography_path, allow_pickle=True)[0]

    elif method == "DSM":
        # load OrthophotoTiffile
        orthophoto_win_tif_obj, __ = DSM.load_orthophoto(args.OrthoPhotoTif)
        # creat/load raster
        if not os.path.exists(args.ToGroundRaster):
            coords = DSM.load_dsm_points(args)
            intrinsic_dict = DSM.load_json_file(args.INTRINSICS_PATH)
            projection_dict = DSM.load_json_file(args.EXTRINSICS_PATH)
            DSM.create_raster(args, args.MetaData["camera"], coords, projection_dict, intrinsic_dict)
        
        with open(args.ToGroundRaster, 'rb') as f:
            GroundRaster = DSM.pickle.load(f)

    with open(out_path, 'w') as out_file:
        for index, row in tqdm(df.iterrows(), total=len(df)):
            # select contact point
            x, y = get_contact_point(row, args)
            # reproject contact point
            new_point = reproj_point(args, x, y, method, M=M, GroundRaster=GroundRaster, TifObj = orthophoto_win_tif_obj)
            # complete the new entry
            new_entry = ""
            for col in df.columns:
                new_entry += f"{row[col]},"
            new_entry += f"{x},"
            new_entry += f"{y},"
            new_entry += f"{new_point[0]},"
            new_entry += f"{new_point[1]}" # last value does not have the ","
            print(new_entry, file=out_file)

def reproject(args, source, method, from_back_up = False):
    # try to reproject tracking results
    # method can be "Homography" / "DSM"
    if source == "tracks":
        out_path = args.ReprojectedPoints 

        if from_back_up:
            df = pd.read_pickle(args.TrackingPklBackUp)
        else:
            df = pd.read_pickle(args.TrackingPkl)
        
        reproject_df(args, df, out_path, method)
        store_to_pickle(args)
        return SucLog("Backprojection executed successfully from tracks")

    elif source == "detections":
        out_path = args.ReprojectedPointsForDetection

        if from_back_up:
            df = pd.read_pickle(args.DetectionPklBackUp)
        else:
            df = pd.read_pickle(args.DetectionPkl)
        
        reproject_df(args, df, out_path, method)
        store_to_pickle_to_detection(args)
        return SucLog("Backrojection executed successfully from detections")
    else:
        return FailLog(f"{source} not a valid backprojection source")

def store_to_pickle(args):
    df = reprojected_df(args)
    df.to_pickle(args.ReprojectedPkl)

def store_to_pickle_to_detection(args):
    df = reprojected_df_for_detection(args)
    df.to_pickle(args.ReprojectedPklForDetection)

def reprojected_df_for_detection(args):
    in_path = args.ReprojectedPointsForDetection
    df = pd.read_pickle(args.DetectionPkl)
    # reprojected df will have all the columns of detection df 
    # with addition of xcp, ycp, x, y
    # which are contact point coordinates and  backprojected coordinates
    points = []
    with open(in_path, "r") as f_in:
        lines = f_in.readlines()
    for line in lines:
        cells = list(map(float, line.split(",")[-4:]))
        points.append(cells)
    points = np.array(points)
    
    data  = {}
    for i, col in enumerate(df.columns):
        data[col] = df[col]

    data["y"]   = points[:, -1]
    data["x"]   = points[:, -2]
    data["ycp"] = points[:, -3]
    data["xcp"] = points[:, -4]

    return pd.DataFrame.from_dict(data)

def reprojected_df(args):
    in_path = args.ReprojectedPoints
    df = pd.read_pickle(args.TrackingPkl)
    # reprojected df will have all the columns of tracking df 
    # with addition of xcp, ycp, x, y
    # which are contact point coordinates and  backprojected coordinates 
    points = np.loadtxt(in_path, delimiter=',')
    data  = {}
    for i, col in enumerate(df.columns):
        data[col] = points[:, i]

    data["y"]   = points[:, -1]
    data["x"]   = points[:, -2]
    data["ycp"] = points[:, -3]
    data["xcp"] = points[:, -4]

    return pd.DataFrame.from_dict(data)

def vishomographygui(args):
    if not os.path.exists(args.HomographyTopView):
        print(ProcLog("intersection topview is not given; will fetch from gmaps"))
        download_top_view_image(args)
        
    if not os.path.exists(args.HomographyStreetView):
        print(ProcLog("intersection streetview is not given; will choose videos first frame"))
        save_frame_from_video(args)

    first_image_path = args.HomographyStreetView
    second_image_path = args.HomographyTopView
    homography_path = args.HomographyNPY
    save_path = args.VisHomographyPth

    img1 = cv.imread(first_image_path)
    img2 = cv.imread(second_image_path)
    rows1, cols1, dim1 = img1.shape
    rows2, cols2, dim2 = img2.shape
    

    M = np.load(homography_path, allow_pickle=True)[0]

    img12 = cv.warpPerspective(img1, M, (cols2, rows2))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    ax1.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
    ax1.set_title("camera view")
    ax2.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
    ax2.set_title("top view")

    ax3.imshow(cv.cvtColor(img12, cv.COLOR_BGR2RGB))
    ax3.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB), alpha=0.3)
    ax3.set_title("camera view reprojected on top view")
    plt.savefig(save_path, bbox_inches='tight')

    plt.close("all")
    # save projected pixels seperately. 
    projection_save_path = save_path[:-3] + "allpix.png"
    plt.imshow(cv.cvtColor(img12, cv.COLOR_BGR2RGB))
    plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB), alpha=0.3)
    plt.axis("off")
    plt.savefig(projection_save_path, bbox_inches='tight')

    # save a picture for selected points correspondance
    point_pairs_plot(args)

    return SucLog("Vis Homography executed successfully") 

def point_pairs_plot(args):
    first_image_path = args.HomographyStreetView
    second_image_path = args.HomographyTopView
    homography_path = args.HomographyNPY
    points_path = args.HomographyCSV

    img1 = cv.imread(first_image_path)
    img2 = cv.imread(second_image_path)

    points = pd.read_csv(points_path)

    rows1, cols1, dim1 = img1.shape
    rows2, cols2, dim2 = img2.shape

    M = np.load(homography_path, allow_pickle=True)[0]
    Mp = np.linalg.inv(M)

    # plot on top view
    for i , row in tqdm(points.iterrows()):
        x, y = row[2], row[3]
        img2 = cv.circle(img2, (x,y), radius=4, color=(255, 0, 0), thickness=3)

    for i, row in points.iterrows():
        x, y = row[0], row[1]
        point = np.array([x, y, 1])
        new_point = M.dot(point)
        new_point /= new_point[2]
        xp, yp = int(new_point[0]), int(new_point[1]) 
        img2 = cv.circle(img2, (xp,yp), radius=4, color=(0, 0, 255), thickness=3)

    plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
    save_path = args.VisHomographyPth[:-3] + "paripoints.top.png"
    plt.savefig(save_path, bbox_inches='tight')

    # plot on camera view
    for i , row in tqdm(points.iterrows()):
        x, y = row[0], row[1]
        img1 = cv.circle(img1, (x,y), radius=20, color=(255, 0, 0), thickness=10)

    for i, row in points.iterrows():
        x, y = row[2], row[3]
        point = np.array([x, y, 1])
        new_point = Mp.dot(point)
        new_point /= new_point[2]
        xp, yp = int(new_point[0]), int(new_point[1]) 
        img1 = cv.circle(img1, (xp,yp), radius=20, color=(0, 0, 255), thickness=10)

    plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
    save_path = args.VisHomographyPth[:-3] + "paripoints.street.png"
    plt.savefig(save_path, bbox_inches='tight')

def vis_reprojected_tracks(args):
    first_image_path = args.HomographyStreetView
    second_image_path = args.HomographyTopView
    
    cam_df = pd.read_pickle(args.TrackingPkl)
    ground_df = pd.read_pickle(args.ReprojectedPkl)
    save_path = args.PlotAllTrajPth

    # tracks_path = "./../Results/GX010069_tracking_sort.txt"
    # transformed_tracks_path = "./../Results/GX010069_tracking_sort_reprojected.txt"

    # tracks = np.loadtxt(tracks_path, delimiter=",")
    # transformed_tracks = np.loadtxt(transformed_tracks_path, delimiter=",")

    img1 = cv.imread(first_image_path)
    img2 = cv.imread(second_image_path)
    rows1, cols1, dim1 = img1.shape
    rows2, cols2, dim2 = img2.shape

    alpha=0.6
    M = np.load(args.HomographyNPY, allow_pickle=True)[0]
    img12 = cv.warpPerspective(img1, M, (cols2, rows2))
    img2 = cv.addWeighted(img2, alpha, img12, 1 - alpha, 0)

    unique_track_ids = np.unique(cam_df['id'])
    # M = np.load(homography_path, allow_pickle=True)[0]
    # img12 = cv.warpPerspective(img1, M, (cols2, rows2))
    for  track_id in tqdm(unique_track_ids):
        # mask = tracks[:, 1]==track_id
        # tracks_id = tracks[mask]
        # if len(tracks_id) < 40: continue
        # transformed_tracks_id = transformed_tracks[mask]
        cam_df_id = cam_df[cam_df['id']==track_id]
        ground_df_id = ground_df[ground_df['id']==track_id]
        
        c = 0
        for i, row in cam_df_id.iterrows():
            x, y = int((row['x2'] + row['x1'])/2), int(row['y2'])
            img1 = cv.circle(img1, (x,y), radius=4, color=(int(c/len(cam_df_id)*255), 70, int(255 - c/len(cam_df_id)*255)), thickness=3)
            c+=1

        c=0
        for i, row in ground_df_id.iterrows():
            x, y = int(row['x']), int(row['y'])
            img2 = cv.circle(img2, (x,y), radius=1, color=(int(c/len(ground_df_id)*255), 70, int(255 - c/len(ground_df_id)*255)), thickness=1)
            c+=1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
    ax2.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
    plt.savefig(save_path)
    return SucLog("Plotting all trajectories execuded")

def vis_contact_point(args):
    # parse detection df using detector module
    det_bp_df = pd.read_pickle(args.ReprojectedPklForDetection)

    # open the original video and process it
    cap = cv2.VideoCapture(args.Video)

    if (cap.isOpened()== False): 
        return FailLog("Error opening video stream or file")
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    out_cap = cv2.VideoWriter(args.VisContactPointPth,cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width,frame_height))

    if not args.ForNFrames is None:
        frames = args.ForNFrames
    for frame_num in tqdm(range(frames)):
        if (not cap.isOpened()):
            return FailLog("Error reading the video")
        ret, frame = cap.read()
        if not ret: continue
        for i, row in det_bp_df[det_bp_df["fn"]==frame_num].iterrows():
            #TODO set up a general condition for all 3D detectors
            if args.Detector == "GTHW73D":
                frame = Detect.draw_3Dbox_on_image(frame , row)
            else:
                frame = Detect.draw_box_on_image(frame, row.x1, row.y1, row.x2, row.y2)
            frame = Detect.draw_point_on_image(frame, row.xcp, row.ycp)
        out_cap.write(frame)

    cap.release()
    out_cap.release()
    return SucLog("sucessfully viz-ed contact points")

def vis_contact_point_top(args):
    
    save_path = args.VisContactPointTopPth

    cap = cv2.VideoCapture(args.Video)
    # Check if camera opened successfully
    if (cap.isOpened()== False): return FailLog("could not open input video")

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    second_image_path = args.HomographyTopView
    img2 = cv.imread(second_image_path)
    rows2, cols2, dim2 = img2.shape
    frame_width , frame_height  = cols2 , rows2

    out_cap = cv2.VideoWriter(save_path,cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width,frame_height))
    color = (0, 0, 102)

    if not args.ForNFrames is None:
        frames = int(min(frames, args.ForNFrames))

    df = pd.read_pickle(args.ReprojectedPklForDetection)

    for frame_num in tqdm(range(frames)):
        img2_fn = copy.deepcopy(img2)
        df_fn = df[df.fn==(frame_num)]
        for i, row in df_fn.iterrows():

            img2_fn = Detect.draw_point_on_image(img2_fn, row.x, row.y)

        out_cap.write(img2_fn)
    out_cap.release()
    return SucLog("sucessfully viz-ed contact points on top view")

def vis_contact_point_top_mc(args, args_mc):
    args_mc_flat = flatten_args(args_mc)
    save_path = args.VisContactPointTopPth

    # get number of frames and fps from one of the videos
    cap = cv2.VideoCapture(args_mc_flat[0].Video)
    if (cap.isOpened()== False): return FailLog("could not open input video")

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    # get top view from one of the videos
    second_image_path = args_mc_flat[0].HomographyTopView
    img2 = cv.imread(second_image_path)
    rows2, cols2, dim2 = img2.shape
    frame_width , frame_height  = cols2 , rows2

    out_cap = cv2.VideoWriter(save_path,cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width,frame_height))
    color = (0, 0, 102)

    if not args.ForNFrames is None:
        frames = int(min(frames, args.ForNFrames))

    # load backprojected detections
    dfs = []
    for arg_i in args_mc_flat:
        df = pd.read_pickle(arg_i.ReprojectedPklForDetection)
        dfs.append(df)

    # plot points on each frame
    for frame_num in tqdm(range(frames)):
        img2_fn = copy.deepcopy(img2)
        for df in dfs:
            df_fn = df[df.fn==(frame_num)]
            for i, row in df_fn.iterrows():
                img2_fn = Detect.draw_point_on_image(img2_fn, row.x, row.y)

        out_cap.write(img2_fn)
    out_cap.release()
    return SucLog("sucessfully viz-ed MC contact points on top view")

def eval_contact_points_ms(arg_top, args_mcss):
    args_mcss_flat = flatten_args(args_mcss)
    # gatther all distances
    distances = []
    distances_top  = []

    for args in tqdm(args_mcss_flat):
        # gett 2D GT args
        # get 3D GT args
        args_gt_2d = get_args_gt(args)
        args_gt_3d = get_args_gt3D(args)

        # read backprojected detections + contact points
        df       = pd.read_pickle(args.ReprojectedPklForDetection)
        df_gt_2d = pd.read_pickle(args_gt_2d.ReprojectedPklForDetection)
        df_gt_3d = pd.read_pickle(args_gt_3d.ReprojectedPklForDetection)

        # if detector is a 3D detector, modify x1, y1, x2, y2
        # to x2D1, y2D1, x2D2, y2D2 so that it is consistent with
        # a 2D detector
        if "x2D1" in df.columns:
            df.x1 = df.x2D1
            df.y1 = df.y2D1
            df.x2 = df.x2D2
            df.y2 = df.y2D2

        # get unique frames based on annotations
        unique_frames = np.unique(df_gt_3d.fn)

        # for each frame in annotations
        for fn in unique_frames:
            # get the detections for frame "fn"
            df_fn       = df[df.fn == fn]
            df_gt_2d_fn = df_gt_2d[df_gt_2d.fn == fn]
            df_gt_3d_fn = df_gt_3d[df_gt_3d.fn == fn]

            # use 2D args for matching 2D detections
            for i, row in df_fn.iterrows():
                gt_2d_match_row = get_det_with_max_iou(row, df_gt_2d_fn)
                if gt_2d_match_row is None: #could not match
                    continue
                # match with uuid/id
                gt_3d_match_row = df_gt_3d_fn[df_gt_3d_fn.uuid == gt_2d_match_row.uuid]

                # first check if both 2d and 3d gt are available
                # if 3d and 2d are not available at the same time we will just skip the detection
                if len(gt_3d_match_row) == 0:
                    continue
                
                # compute distance on image
                cp_est    = np.array([row.xcp, row.ycp])
                cp_gt     = np.array([gt_3d_match_row.xcp.iloc[0], gt_3d_match_row.ycp.iloc[0]])
                dist_row  = np.linalg.norm(cp_est - cp_gt)
                distances.append(dist_row)

                # compute distance on ground
                cp_est_top    = np.array([row.x, row.y])
                cp_gt_top     = np.array([gt_3d_match_row.x.iloc[0], gt_3d_match_row.y.iloc[0]])
                dist_row_top  = np.linalg.norm(cp_est_top - cp_gt_top)
                distances_top.append(dist_row_top)


    cp_error_image = np.mean(distances)
    cp_error_top   = np.mean(distances_top)
    print(cp_error_image, cp_error_top)

    df_data = {}
    df_data["CPErrorImage"] = [cp_error_image]
    df_data["CPErrorTop"]   = [cp_error_top]
    df = pd.DataFrame.from_dict(df_data)
    df.to_csv(arg_top.CPError, index=False)

    return SucLog("evaluated CP selection MS")

def eval_contact_points(args):
    # get 2D GT args
    # get 3D GT args
    args_gt_2d = get_args_gt(args)
    args_gt_3d = get_args_gt3D(args)

    # read backprojected detections + contact points
    df       = pd.read_pickle(args.ReprojectedPklForDetection)
    df_gt_2d = pd.read_pickle(args_gt_2d.ReprojectedPklForDetection)
    df_gt_3d = pd.read_pickle(args_gt_3d.ReprojectedPklForDetection)

    # if detector is a 3D detector, modify x1, y1, x2, y2
    # to x2D1, y2D1, x2D2, y2D2 so that it is consistent with
    # a 2D detector
    if "x2D1" in df.columns:
        df.x1 = df.x2D1
        df.y1 = df.y2D1
        df.x2 = df.x2D2
        df.y2 = df.y2D2

    # get unique frames based on annotations
    unique_frames = np.unique(df_gt_3d.fn)


    # gatther all distances
    distances      = []
    distances_top  = []

    # for each frame in annotations
    for fn in tqdm(unique_frames):
        # get the detections for frame "fn"
        df_fn       = df[df.fn == fn]
        df_gt_2d_fn = df_gt_2d[df_gt_2d.fn == fn]
        df_gt_3d_fn = df_gt_3d[df_gt_3d.fn == fn]

        # use 2D args for matching 2D detections
        for i, row in df_fn.iterrows():
            gt_2d_match_row = get_det_with_max_iou(row, df_gt_2d_fn)
            if gt_2d_match_row is None: #could not match
                continue
            # match with uuid/id
            gt_3d_match_row = df_gt_3d_fn[df_gt_3d_fn.uuid == gt_2d_match_row.uuid]

            # first check if both 2d and 3d gt are available
            # if 3d and 2d are not available at the same time we will just skip the detection
            if len(gt_3d_match_row) == 0:
                continue

            # image pixel error
            cp_est    = np.array([row.xcp, row.ycp])
            cp_gt     = np.array([gt_3d_match_row.xcp.iloc[0], gt_3d_match_row.ycp.iloc[0]])
            dist_row  = np.linalg.norm(cp_est - cp_gt)
            distances.append(dist_row)

            # top view pixel error(convertable to meter)
            cp_est_top    = np.array([row.x, row.y])
            cp_gt_top     = np.array([gt_3d_match_row.x.iloc[0], gt_3d_match_row.y.iloc[0]])
            dist_row_top  = np.linalg.norm(cp_est_top - cp_gt_top)
            distances_top.append(dist_row_top)
            
            
    cp_error_image = np.mean(distances)
    cp_error_top   = np.mean(distances_top)
    print(cp_error_image, cp_error_top)
    # save this number somewhere

    df_data = {}
    df_data["CPErrorImage"] = [cp_error_image]
    df_data["CPErrorTop"]   = [cp_error_top]
    df = pd.DataFrame.from_dict(df_data)
    df.to_csv(args.CPError, index=False)

    return SucLog("evaluated CP selection")

def eval_cp_MC(arg_top, args_mc):
    args_mc_flat = flatten_args(args_mc)

    # group 3D coordinates after matching with cuboids
    uuid_to_xy = {}
    list_args_gt_2d = []
    list_args_gt_3d = []
    list_df         = []
    list_df_gt_2d   = []
    list_df_gt_3d   = []

    list_frames = []
    list_ids    = []

    for args in args_mc_flat:
        # get 2D GT args
        # get 3D GT args
        args_gt_2d = get_args_gt(args)
        args_gt_3d = get_args_gt3D(args)

        # read backprojected detections + contact points
        df       = pd.read_pickle(args.ReprojectedPklForDetection)
        df_gt_2d = pd.read_pickle(args_gt_2d.ReprojectedPklForDetection)
        df_gt_3d = pd.read_pickle(args_gt_3d.ReprojectedPklForDetection)

        # if detector is a 3D detector, modify x1, y1, x2, y2
        # to x2D1, y2D1, x2D2, y2D2 so that it is consistent with
        # a 2D detector
        if "x2D1" in df.columns:
            df.x1 = df.x2D1
            df.y1 = df.y2D1
            df.x2 = df.x2D2
            df.y2 = df.y2D2
        

        # get unique frames based on annotations
        unique_frames = np.unique(df_gt_3d.fn).tolist()
        unique_ids    = np.unique(df_gt_3d.id).tolist()


        list_args_gt_2d.append(args_gt_2d)
        list_args_gt_3d.append(args_gt_3d)
        list_df.append(df)
        list_df_gt_2d.append(df_gt_2d)
        list_df_gt_3d.append(df_gt_3d)
        list_frames.extend(unique_frames)
        list_ids.extend(unique_ids)

    unique_frames = np.unique(list_frames)  
    unique_ids = np.unique(list_ids)

    # for each frame in annotations
    for fn in tqdm(unique_frames[:1]):
        for id in tqdm(unique_ids):
        # raise NotImplementedError
        # get the detections for frame "fn"
            df_fn       = df[df.fn == fn]
            df_gt_2d_fn = df_gt_2d[df_gt_2d.fn == fn]
            df_gt_3d_fn = df_gt_3d[df_gt_3d.fn == fn]

            # use 2D args for matching 2D detections
            for i, row in df_fn.iterrows():
                gt_2d_match_row = get_det_with_max_iou(row, df_gt_2d_fn)
                if gt_2d_match_row is None: #could not match
                    continue
                # match with uuid/id
                gt_3d_match_row = df_gt_3d_fn[df_gt_3d_fn.id == gt_2d_match_row.id]

                # first check if both 2d and 3d gt are available
                # if 3d and 2d are not available at the same time we will just skip the detection
                if len(gt_3d_match_row) == 0:
                    continue

                # top view pixel error(convertable to meter)
                cp_est_top    = [row.x, row.y]

                id = df_gt_3d_fn.id.iloc[0]

                if not (fn, id) in uuid_to_xy:
                    uuid_to_xy[(fn, id)] = []
            
                uuid_to_xy[(fn, id)].append(cp_est_top)    
            
    # for each uuid, if there are more than one 3D coor compute average dist
    def pairwise_distance_np(coordinates):
        # Convert the list of coordinates to a NumPy array
        points = np.array(coordinates)
        # Calculate the difference matrix along each dimension
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        # Compute the squared distances for each pair
        squared_dist = np.sum(diff**2, axis=2)
        # Take the square root to get the Euclidean distances
        distances = np.sqrt(squared_dist)
        # Since the distance matrix is symmetric, we take the upper triangle (excluding the diagonal) to avoid double counting
        upper_triangle_indices = np.triu_indices_from(distances, k=1)
        # Compute the average distance
        average_distance = np.mean(distances[upper_triangle_indices])
        return average_distance

    print(uuid_to_xy)
    return

    distances_top = []
    for key in tqdm(uuid_to_xy):
        if len(uuid_to_xy[key]) > 1: # if box is visible >=2 cams
            distances_top.append(pairwise_distance_np(uuid_to_xy[key]))

    cp_error_top = np.mean(distances_top)

    df_data = {}
    df_data["CPErrorMC"]   = [cp_error_top]
    df = pd.DataFrame.from_dict(df_data)
    df.to_csv(arg_top.CPErrorMC, index=False)

    return cp_error_top, len(distances_top)

def eval_cp_MC_MS(args, args_ms, args_mcs):
    errors , supports = [], []
    for arg_ms, arg_mc in tqdm(zip(args_ms, args_mcs), total=len(args_ms)):
        error, support = eval_cp_MC(arg_ms, arg_mc)
        errors.append(error)
        supports.append(support)
    
    errors = np.array(errors)
    supports = np.array(supports)

    top_error = np.sum(errors * supports) / np.sum(supports)
    print(top_error)

    df_data = {}
    df_data["CPErrorMC"]   = [top_error]
    df = pd.DataFrame.from_dict(df_data)
    df.to_csv(args.CPErrorMC, index=False)

    return SucLog("evaluated CP selection MC MS")