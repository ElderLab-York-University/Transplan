# this file is developed to envoce tracklabeling GUI
from Utils import * 
from Libs import *

# args variables for tracklabelling gui
    # args.TrackLabellingExportPth
    # args.VisLabelledTracksPth

def tracklabelinggui(args):
    export_path = os.path.abspath(args.TrackLabellingExportPth)
    cwd = os.getcwd()
    os.chdir("./cluster_labelling_gui/")
    ret = os.system(f"sudo python3 cam_gen.py --Export='{export_path}'")
    os.chdir(cwd)

    if ret==0:
        return SucLog("track labelling executed successfully")
    return FailLog("track labelling ended with non-zero return value")


def vis_labelled_tracks(args):
    save_path = args.VisLabelledTracksPth
    tracks = pd.read_pickle(args.TrackLabellingExportPth)
    tracks = tracks.sort_values("moi")
    second_image_path = args.HomographyTopView

    img2 = cv.imread(second_image_path)
    rows2, cols2, dim2 = img2.shape
    for i in range(len(tracks)):
        track = tracks.iloc[i]
        traj = track['trajectory']
        moi = track["moi"]
        for j , p in enumerate(traj):
            x , y = int(p[0]), int(p[1])
            c = moi_color_dict[moi]
            img2 = cv.circle(img2, (x,y), radius=2, color=c, thickness=2)

    plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
    plt.savefig(save_path)

    return SucLog("labeled trackes plotted successfully")


