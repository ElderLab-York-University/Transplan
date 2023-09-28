from Utils import *
from Libs import *
import Track
import Maps
import DSM
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
    os.system(f"sudo python3 main.py --StreetView='{street}' --TopView='{top}' --Txt='{txt}' --Npy='{npy}' --Csv='{csv}'")
    os.chdir(cwd)

def reproj_point(args, x, y, method, **kwargs):
    if method == "Homography":
        return reproj_point_homography(x, y, kwargs["M"])
        
    elif method == "DSM":
        return reproj_point_dsm(x, y, kwargs["GroundRaster"], kwargs["TifObj"])
    
    else: raise NotImplemented

def reproj_point_homography(x, y, M):
    point = np.array([x, y, 1])
    new_point = M.dot(point)
    new_point /= new_point[2]
    return new_point

def reproj_point_dsm(x, y, img_ground_raster, orthophoto_win_tif_obj):
    u_int, v_int = int(x), int(y)
    # mathced_coord are the real-world coordinates we need to change them to top-view pixel coordinates(for consistancy reasons)
    matched_coord =  img_ground_raster[v_int, u_int]
    orthophoto_proj_idx = orthophoto_win_tif_obj.index(*matched_coord[:-1])
    # pass (orthophoto_proj_idx[1], orthophoto_proj_idx[0]) to csv.drawMarker
    return orthophoto_proj_idx

def reproject_df(args, df, out_path, method):
    # we will load M and Raster here and pass it on to speed up the projection
    M = None
    GroundRaster = None
    orthophoto_win_tif_obj, __ = DSM.load_orthophoto(args.OrthoPhotoTif)

    if method == "Homography":
        homography_path = args.HomographyNPY
        M = np.load(homography_path, allow_pickle=True)[0]

    elif method == "DSM":
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
            x, y = (row['x2']+row['x1'])/2, row['y2']
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
    # method can be "Homography" / "DTM"
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
    plt.savefig(save_path)

    return SucLog("Vis Homography executed successfully") 

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
