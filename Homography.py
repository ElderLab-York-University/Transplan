from Utils import *
from Libs import *
from Track import *
# import homographygui.tabbed_ui_func as tui

#hints the vars set for homography are 
    # args.HomographyStreetView
    # args.HomographyTopView
    # args.HomographyTXT
    # args.HomographyNPY
    # args.HomographyCSV
    # args.ReprojectedPoints 

def homographygui(args):
    # assume homography repo is made in results
    # check if homography pair pictures are available with the video
    if not os.path.exists(args.HomographyTopView):
        return FailLog(f"intersection top view view is missing {args.HomographyTopView}")
    if not os.path.exists(args.HomographyStreetView):
        print(ProcLog("intersection streetview is not given; will choose videos first frame"))
        save_frame_from_video(args.Video, args.HomographyStreetView)

    # lunch homography gui
    lunch_homographygui(args)
    return SucLog("Homography GUI executed successfully")    
    # if all good lunch homographGUI

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
    df = current_tracker.df(args)
    
    M = np.load(homography_path, allow_pickle=True)[0]
    with open(out_path, 'w') as out_file:
        for index, row in tqdm(df.iterrows()):
            # fn, idd, x, y = track[0], track[1], (track[2] + track[4]/2), (track[3] + track[5])/2
            fn, idd, x, y = row['fn'], row['id'], row['x2'], (row['y1'] + row['y2'])/2
            point = np.array([x, y, 1])
            new_point = M.dot(point)
            new_point /= new_point[2]
            print(f'{int(fn)},{int(idd)},{new_point[0]},{new_point[1]}', file=out_file)
    return SucLog("Homography reprojection executed successfully")   

def reprojected_df(args):
    in_path = args.ReprojectedPoints 
    points = np.loadtxt(in_path, delimiter=',')
    data  = {}
    data["fn"] = points[:, 0]
    data["id"] = points[:, 1]
    data["x"]  = points[:, 2]
    data["y"]  = points[:, 3]
    return pd.DataFrame.from_dict(data)

