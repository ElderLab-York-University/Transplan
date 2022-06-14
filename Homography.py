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
    #args.VisHomographyPth

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