# this file is developed to envoce tracklabeling GUI
from Utils import * 
from Libs import *
from counting.resample_gt_MOI.resample_typical_tracks import track_resample

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


def extract_common_tracks(args):
    tracks_path = args.ReprojectedPkl
    tracks_meter_path = args.ReprojectedPklMeter
    top_image = args.HomographyTopView
    meta_data = args.MetaData # dict is already loaded
    HomographyNPY = args.HomographyNPY
    exportpath = args.TrackLabellingExportPth
    moi_clusters = args.MetaData["moi_clusters"]

    # load data
    M = np.load(HomographyNPY, allow_pickle=True)[0]
    tracks = group_tracks_by_id(pd.read_pickle(tracks_path))
    tracks_meter = group_tracks_by_id(pd.read_pickle(tracks_meter_path))
    tracks['index_mask'] = tracks_meter['trajectory'].apply(lambda x: track_resample(x, return_mask=True))
    img = plt.imread(top_image)
    img1 = cv.imread(top_image)

    # create roi polygon
    roi_rep = []
    for p in args.MetaData["roi"]:
        point = np.array([p[0], p[1], 1])
        new_point = M.dot(point)
        new_point /= new_point[2]
        roi_rep.append([new_point[0], new_point[1]])

    pg = MyPoly(roi_rep)
    th = args.MetaData["roi_percent"] * np.sqrt(pg.area)

    counter = 0
    mask = []
    i_strs = []
    i_ends = []
    moi = []

    # find and plot proper tracks(complete and monotonic)
    for i, row in tqdm(tracks.iterrows(), total=len(tracks)):
        traj = row["trajectory"]
        d_str, i_str = pg.distance(traj[0])
        d_end, i_end = pg.distance(traj[-1])
        i_strs.append(i_str)
        i_ends.append(i_end)
        moi.append(str_end_to_moi(i_str, i_end))

        if (d_str < th) and (d_end < th) and (not i_str == i_end) and is_monotonic(traj[row["index_mask"]]):
            mask.append(True)
            counter += 1
            c=0
            for x, y in traj:
                x, y = int(x), int(y)
                img1 = cv.circle(img1, (x,y), radius=1, color=(int(c/len(traj)*255), 70, int(255 - c/len(traj)*255)), thickness=1)
                c+=1
        else:
            mask.append(False)

    print(counter/len(tracks))
    plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
    plt.show()

    # temporarily add info to track dataframe
    tracks['i_str'] = i_strs
    tracks['i_end'] = i_ends
    tracks['moi'] = moi
    tracks_meter['i_str'] = i_strs
    tracks_meter['i_end'] = i_ends 
    tracks_meter['moi'] = moi
    # plt.hist(tracks[mask]['moi'])
    # plt.show()


    # extract all starts and ends from proper tracks
    starts = []
    ends=[]
    for i, row in tracks[mask].iterrows():
        starts.append(row['trajectory'][0])
        ends.append(row['trajectory'][-1])

    starts = np.array(starts)
    ends = np.array(ends)
    plt.imshow(img)
    plt.scatter(starts[:, 0],starts[:, 1], alpha=0.3)
    plt.show()
    

    # cluster tracks and choose common tracks as cluster centers
    chosen_indexes = []
    for mi in moi_clusters.keys():
        tracks_mi = tracks[mask][tracks[mask]['moi']==mi]
        index_mi = tracks_mi.index
        starts = []
        ends=[]
        for i, row in tracks_mi.iterrows():
            starts.append(row['trajectory'][0])
            ends.append(row['trajectory'][-1])
        starts = np.array(starts)
        ends = np.array(ends)
        num_cluster = moi_clusters[mi]
        if len(starts) < 1:
            print(f"moi {mi} was empty")
            continue

        clt = sklearn.cluster.KMeans(n_clusters = num_cluster)
        clt.fit(starts)
        c_start = clt.predict(starts)
        p_start = np.array([clt.score(x.reshape(1, -1)) for x in starts])
        cluster_labels = np.unique(c_start)
        plt.imshow(img)
        plt.scatter(starts[:, 0], starts[:, 1], c=c_start)
        plt.show()
        for c_label in cluster_labels:
            mask_c = c_start == c_label
            i_c = np.argmax(p_start[mask_c])
            chosen_index = index_mi[mask_c][i_c]
            chosen_indexes.append(chosen_index)

    # plot final common tracks
    img1 = cv.imread(top_image)
    for i, row in tracks.loc[chosen_indexes].iterrows():
        traj = row["trajectory"]
        c=0
        for x, y in traj:
            x, y = int(x), int(y)
            img1 = cv.circle(img1, (x,y), radius=1, color=(int(c/len(traj)*255), 70, int(255 - c/len(traj)*255)), thickness=1)
            c+=1

    plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
    plt.show()

    # save common tracks as labelled tracks
    tracks_labelled = group_tracks_by_id(pd.read_pickle(tracks_path))
    tracks_labelled["moi"] = tracks["moi"].apply(lambda x: int(x))
    tracks_labelled = tracks_labelled.loc[chosen_indexes]
    tracks_labelled.to_pickle(exportpath)
    return SucLog("common trakces extracted successfully")



class MyPoly():
    def __init__(self, roi):
        self.poly = sympy.Polygon(*roi)
        self.lines = []
        for i in range(len(roi)-1):
            self.lines.append(sympy.Line(sympy.Point(roi[i]), sympy.Point(roi[i+1])))
        self.lines.append(sympy.Line(sympy.Point(*roi[-1]), sympy.Point(*roi[0])))
        
    def distance(self, point):
        p = sympy.Point(*point)
        distances = []
        for line in self.lines:
            distances.append(float(line.distance(p)))
        distances = np.array(distances)
        min_pos = np.argmin(distances)
        return distances[min_pos], int(min_pos)

    @property
    def area(self):
        return float(self.poly.area)

def is_monotonic(traj):
    orgin = traj[0]
    max_distance = -1
    for p in traj:
        dp = np.linalg.norm(p - orgin)
        if dp < max_distance: return False
        max_distance = dp
    return True

def group_tracks_by_id(df):
    # this function was writtern for grouping the tracks with the same id
    # usinig this one can load the data from a .txt file rather than .mat file
    all_ids = np.unique(df['id'].to_numpy(dtype=np.int64))
    data = {"id":[], "trajectory":[], "frames":[]}
    for idd in tqdm(all_ids):
        frames = df[df['id']==idd]["fn"].to_numpy(np.float32)
        id = idd
        trajectory = df[df['id']==idd][["x", "y"]].to_numpy(np.float32)
        
        data["id"].append(id)
        data["frames"].append(frames)
        data["trajectory"].append(trajectory)
    df2 = pd.DataFrame(data)
    return df2


def str_end_to_moi(str, end):
    str_end_moi = {}
    str_end_moi[(3, 0)] = '1'
    str_end_moi[(3, 1)] = '2'
    str_end_moi[(3, 2)] = '3'
    str_end_moi[(2, 3)] = '4'
    str_end_moi[(2, 0)] = '5'
    str_end_moi[(2, 1)] = '6'
    str_end_moi[(1, 2)] = '7'
    str_end_moi[(1, 3)] = '8'
    str_end_moi[(1, 0)] = '9'
    str_end_moi[(0, 1)] = '10'
    str_end_moi[(0, 2)] = '11'
    str_end_moi[(0, 3)] = '12'
    if (str ,end) in str_end_moi:
        return str_end_moi[(str, end)]
    return -1