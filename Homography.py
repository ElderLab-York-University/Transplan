from Utils import *
from Libs import *
from Track import *
import Maps
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
        save_frame_from_video(args.Video, args.HomographyStreetView)
        

    # lunch homography gui
    lunch_homographygui(args)
    return SucLog("Homography GUI executed successfully")    
    # if all good lunch homographGUI

def download_top_view_image(args):
    center = args.MetaData['center']
    file_name = args.HomographyTopView
    Maps.download_image(center=center, file_name=file_name)

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

def reproject(args):
    homography_path = args.HomographyNPY
    out_path = args.ReprojectedPoints 

    current_tracker = trackers[args.Tracker]
    # df = current_tracker.df(args)
    df = pd.read_pickle(args.TrackingPklBackUp)
    
    M = np.load(homography_path, allow_pickle=True)[0]
    with open(out_path, 'w') as out_file:
        for index, row in tqdm(df.iterrows(), total=len(df)):
            # fn, idd, x, y = track[0], track[1], (track[2] + track[4]/2), (track[3] + track[5])/2
            fn, idd, x, y = row['fn'], row['id'], row['x2'], (row['y1'] + row['y2'])/2
            point = np.array([x, y, 1])
            new_point = M.dot(point)
            new_point /= new_point[2]
            print(f'{int(fn)},{int(idd)},{new_point[0]},{new_point[1]}', file=out_file)

    store_to_pickle(args)
    return SucLog("Homography reprojection executed successfully")

def store_to_pickle(args):
    df = reprojected_df(args)
    df.to_pickle(args.ReprojectedPkl)   

def reprojected_df(args):
    in_path = args.ReprojectedPoints 
    points = np.loadtxt(in_path, delimiter=',')
    data  = {}
    data["fn"] = points[:, 0]
    data["id"] = points[:, 1]
    data["x"]  = points[:, 2]
    data["y"]  = points[:, 3]
    return pd.DataFrame.from_dict(data)

def vishomographygui(args):
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
            x, y = int(row['x2']), int((row['y1']+row['y2'])/2)
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
