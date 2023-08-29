# this file is developed to envoce tracklabeling GUI
from Utils import * 
from Libs import *
from counting.resample_gt_MOI.resample_typical_tracks import track_resample
import Track

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
    # plot labelled tracks on top view
    alpha = 0.6
    save_path = args.VisLabelledTracksPth
    tracks = pd.read_pickle(args.TrackLabellingExportPth)
    tracks = tracks.sort_values("moi")
    img1 = cv.imread(args.HomographyStreetView)
    img2 = cv.imread(args.HomographyTopView)
    M = np.load(args.HomographyNPY, allow_pickle=True)[0]
    rows2, cols2, dim2 = img2.shape
    img12 = cv.warpPerspective(img1, M, (cols2, rows2))
    img2 = cv.addWeighted(img2, alpha, img12, 1 - alpha, 0)

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
    plt.close("all")

    # plot labelled tracks on image domain as well
    alpha = 0.6
    save_path_street = args.VisLabelledTracksPthImage
    tracks = pd.read_pickle(args.TrackLabellingExportPthImage)
    tracks = tracks.sort_values("moi")
    img1 = cv.imread(args.HomographyStreetView)
    # img2 = cv.imread(args.HomographyTopView)
    M = np.load(args.HomographyNPY, allow_pickle=True)[0]
    rows2, cols2, dim2 = img2.shape
    # img12 = cv.warpPerspective(img1, M, (cols2, rows2))
    # img2 = cv.addWeighted(img2, alpha, img12, 1 - alpha, 0)

    for i in range(len(tracks)):
        track = tracks.iloc[i]
        traj = track['trajectory']
        moi = track["moi"]
        for j , p in enumerate(traj):
            x , y = int(p[0]), int(p[1])
            c = moi_color_dict[moi]
            img2 = cv.circle(img1, (x,y), radius=3, color=c, thickness=3)

    plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
    plt.savefig(save_path_street)
    plt.close("all")

    return SucLog("labeled trackes plotted successfully")

def get_in_roi_points(traj, poly_path):
    mask = []
    for p in traj:
        if poly_path.contains_point(p):
            mask.append(True)
        else:
            mask.append(False)
    return traj[mask]

def get_proper_tracks(args):
    tracks_path = args.ReprojectedPkl
    tracks_meter_path = args.ReprojectedPklMeter
    top_image = args.HomographyTopView
    street_image = args.HomographyStreetView
    # exportpath = args.TrackLabellingExportPth
    meta_data = args.MetaData # dict is already loaded
    HomographyNPY = args.HomographyNPY
    M = np.load(HomographyNPY, allow_pickle=True)[0]
    moi_clusters = {}
    for mi in  args.MetaData["moi_clusters"].keys():
        moi_clusters[int(mi)] = args.MetaData["moi_clusters"][mi]

    img = plt.imread(top_image)
    img1 = cv.imread(top_image)
    img_street = plt.imread(street_image)

    # load data
    tracks = group_tracks_by_id(pd.read_pickle(tracks_path), gp=True)
    tracks_meter = group_tracks_by_id(pd.read_pickle(tracks_meter_path), gp=True)
    tracks['index_mask'] = tracks_meter['trajectory'].apply(lambda x: track_resample(x, return_mask=True, threshold=args.ResampleTH))
    # tracks_meter['trajectory'] = tracks_meter['trajectory'].apply(lambda x: track_resample(x, return_mask=False, threshold=args.ResampleTH))

    # create roi polygon
    roi_rep = []
    for p in args.MetaData["roi"]:
        point = np.array([p[0], p[1], 1])
        new_point = M.dot(point)
        new_point /= new_point[2]
        roi_rep.append([new_point[0], new_point[1]])

    pg = MyPoly(roi_rep, args.MetaData["roi_group"])
    th = args.MetaData["roi_percent"] * np.sqrt(pg.area)
    poly_path = mplPath.Path(np.array(roi_rep))

    # select only tracks that have two interactions with Intersection
    counter = 0
    mask = []
    i_strs = []
    i_ends = []
    p_strs = []
    p_ends = []
    moi = []

    print("!-!: selecting tacks that have at least two interactions with ROI")
    for i, row in tqdm(tracks.iterrows(), total=len(tracks)):
        traj = row["trajectory"][row["index_mask"]]
        int_indexes, int_points = pg.doIntersect(traj, ret_points=True)
        if len(np.unique(int_indexes)) > 1 and is_monotonic(get_in_roi_points(traj, poly_path)): # more than one interactions with ROI
            mask.append(True)
            counter += 1
            i_strs.append(int_indexes[0])
            i_ends.append(int_indexes[-1])
            p_strs.append(int_points[0])
            p_ends.append(int_points[-1])
            moi.append(str_end_to_moi(int_indexes[0], int_indexes[-1]))
            c=0
            for x, y in traj:
                x, y = int(x), int(y)
                img1 = cv.circle(img1, (x,y), radius=1, color=(int(c/len(traj)*255), 70, int(255 - c/len(traj)*255)), thickness=1)
                c+=1
        else:
            mask.append(False)

    print(f"percentage of complete tracks: {counter/len(tracks)}")
    plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
    plt.savefig("multicross.png")
    plt.close("all")

    # modify dfs with mask
    tracks = tracks[mask]
    tracks_meter = tracks_meter[mask]

    # temporarily add info to track dataframe
    tracks['i_str'] = i_strs
    tracks['i_end'] = i_ends
    tracks['moi'] = moi
    tracks['p_str'] = p_strs
    tracks['p_end'] = p_ends

    tracks_meter['i_str'] = i_strs
    tracks_meter['i_end'] = i_ends 
    tracks_meter['moi'] = moi

    return tracks

def cluster_prop_tracks_based_on_str(tracks, args):
    '''
    assuming that tracks has p_str and p_end for their s
    '''
    tracks["clt"] = np.nan
    moi_clusters = {}
    for mi in  args.MetaData["moi_clusters"].keys():
        moi_clusters[int(mi)] = args.MetaData["moi_clusters"][mi]
            # cluster tracks and choose common tracks as cluster centers

    for mi in moi_clusters.keys():
        tracks_mi = tracks[tracks['moi']==mi]
        index_mi = tracks_mi.index
        starts = []
        ends=[]
        for i, row in tracks_mi.iterrows():
            starts.append(row['p_str'])
            ends.append(row['p_end'])
        starts = np.array(starts)
        ends = np.array(ends)
        num_cluster = moi_clusters[mi]
        if len(starts) < 1:
            print(f"moi {mi} was empty")
            continue

        clt = sklearn.cluster.KMeans(n_clusters = min(num_cluster, len(starts)), n_init=100)
        clt.fit(starts)
        c_start = clt.predict(starts)
        tracks.loc[tracks.moi == mi, "clt"] = c_start

    return tracks

def make_uniform_clt_per_moi(tracks, args):
    # sort tracks based on their length(longer comes first)
    tracks_len = []
    for i, row in tracks.iterrows():
        tracks_len.append(len(row.trajectory))
    tracks["traj_len"] = tracks_len
    tracks = tracks.sort_values(by="traj_len", ascending=False)

    # for each moi
    indexes_to_keep = []
    for mi in np.unique(tracks.moi):
        tracks_mi = tracks[tracks['moi']==mi].sort_values(by="traj_len", ascending=False)
        # print(tracks_mi.clt)
        uni_vals, uni_counts = np.unique(tracks_mi.clt, return_counts=True)
        # print(uni_vals)
        # print(uni_counts)
        num_tracks_to_get_from_clt_per_moi = min(np.min(uni_counts), 10)

        print(f"nums to take: {num_tracks_to_get_from_clt_per_moi} moi:{mi} num_clt:{len(uni_vals)} tot_tracks_chosen:{len(uni_vals)*num_tracks_to_get_from_clt_per_moi}")
        for uv in uni_vals:
            idxs_mi_clt  = tracks_mi[tracks_mi.clt == uv].index[:num_tracks_to_get_from_clt_per_moi]
            indexes_to_keep += [i for i in idxs_mi_clt]

    return tracks.loc[indexes_to_keep]


def extract_common_tracks(args):
    tracks_path = args.ReprojectedPkl
    tracks_meter_path = args.ReprojectedPklMeter
    top_image = args.HomographyTopView
    street_image = args.HomographyStreetView
    exportpath = args.TrackLabellingExportPth
    exportpathimage = args.TrackLabellingExportPthImage
    meta_data = args.MetaData # dict is already loaded
    HomographyNPY = args.HomographyNPY
    M = np.load(HomographyNPY, allow_pickle=True)[0]
    moi_clusters = {}
    for mi in  args.MetaData["moi_clusters"].keys():
        moi_clusters[int(mi)] = args.MetaData["moi_clusters"][mi]

    img = plt.imread(top_image)
    img1 = cv.imread(top_image)
    img_street = plt.imread(street_image)

    # create roi polygon
    roi_rep = []
    for p in args.MetaData["roi"]:
        point = np.array([p[0], p[1], 1])
        new_point = M.dot(point)
        new_point /= new_point[2]
        roi_rep.append([new_point[0], new_point[1]])

    pg = MyPoly(roi_rep, args.MetaData["roi_group"])
    th = args.MetaData["roi_percent"] * np.sqrt(pg.area)
    poly_path = mplPath.Path(np.array(roi_rep))

    # get proper tracks
    tracks = get_proper_tracks(args)

    # extract all starts and ends from proper tracks
    starts = []
    ends=[]
    for i, row in tracks.iterrows():
        starts.append(row['p_str'])
        ends.append(row['p_end'])

    starts = np.array(starts)
    ends = np.array(ends)
    plt.imshow(img)
    plt.scatter(starts[:, 0],starts[:, 1], alpha=0.3)
    plt.savefig("multicorss_points.png")
    plt.close("all")

    

    # cluster tracks and choose common tracks as cluster centers
    chosen_indexes = []
    for mi in moi_clusters.keys():
        tracks_mi = tracks[tracks['moi']==mi]
        index_mi = tracks_mi.index
        starts = []
        ends=[]
        for i, row in tracks_mi.iterrows():
            starts.append(row['p_str'])
            ends.append(row['p_end'])
        starts = np.array(starts)
        ends = np.array(ends)
        num_cluster = moi_clusters[mi]
        if len(starts) < 1:
            print(f"moi {mi} was empty")
            continue

        clt = sklearn.cluster.KMeans(n_clusters = min(num_cluster, len(starts)), n_init=100)
        clt.fit(starts)
        c_start = clt.predict(starts)
        p_start = np.array([clt.score(x.reshape(1, -1)) for x in starts])
        cluster_labels = np.unique(c_start)
        plt.imshow(img)
        plt.scatter(starts[:, 0], starts[:, 1], c=c_start)
        plt.savefig(f"int_lane_clt_{mi}.png")
        plt.close("all")

        for c_label in cluster_labels:
            mask_c = c_start == c_label
            i_c = np.argmax(p_start[mask_c])
            chosen_index = index_mi[mask_c][i_c]
            chosen_indexes.append(chosen_index)

    # save common tracks as labelled tracks on ground plane
    tracks_labelled = group_tracks_by_id(pd.read_pickle(tracks_path), gp=True)
    tracks_labelled = tracks_labelled.loc[chosen_indexes]
    tracks_labelled["moi"] = tracks.loc[chosen_indexes]["moi"].apply(lambda x: int(x))
    tracks_labelled.to_pickle(exportpath)

    # save common tracks as labelled tracks on image plane
    tracks_labelled = group_tracks_by_id(pd.read_pickle(tracks_path), gp=False)
    tracks_labelled = tracks_labelled.loc[chosen_indexes]
    tracks_labelled["moi"] = tracks.loc[chosen_indexes]["moi"].apply(lambda x: int(x))
    tracks_labelled.to_pickle(exportpathimage)

    return SucLog("common trakces extracted successfully")

class MyPoly():
    def __init__(self, roi, roi_group):
        self.roi = roi
        self.roi_group = roi_group

        self.poly = sympy.Polygon(*roi)
        self.lines = []
        for i in range(len(roi)-1):
            self.lines.append(sympy.Segment(sympy.Point(roi[i]), sympy.Point(roi[i+1])))
        self.lines.append(sympy.Segment(sympy.Point(*roi[-1]), sympy.Point(*roi[0])))
        
    def distance(self, point):
        p = sympy.Point(*point)
        distances = []
        for line, gp in zip(self.lines, self.roi_group):
            d_line = float(line.distance(p))
            if gp <=0 : d_line = float('inf')
            distances.append(d_line)
        distances = np.array(distances)
        min_pos = np.argmin(distances)
        return distances[min_pos], int(self.roi_group[min_pos])

    def distance_angle_filt(self, p_main, p_second):
        imaginary_line = sympy.Line(sympy.Point(p_main), sympy.Point(p_second))
        min_angle = float('inf')
        min_distance = float('inf')
        min_distance_indx = None
        for i, line in enumerate(self.lines):
            d_main = float(line.distance(p_main))
            d_secn = float(line.distance(p_second))
            if d_secn <  d_main : continue
            else:
                if np.abs(float(imaginary_line.angle_between(line)) - np.pi/2 ) < min_angle:
                    min_distance = float(line.distance(p_main))
                    min_distance_indx  = i
        return min_distance, min_distance_indx
    
    def encloses_point(self, point):
        p = sympy.Point(*point)
        return self.poly.encloses_point(p)

    def doIntersect(self, track, ret_points):
        '''
        hope fully a faster method just to determine intersection indexes
        only returns unique groups that track intersected with
        if re_points=True will return intersection point as well
        '''                    
        int_indexes = []
        int_points = []
        roi_segments = []
        for seg in self.lines:
            roi_segments.append([Point(list(seg.points[0])), Point(list(seg.points[1]))])
        for i in range(len(track)-1):
            if len(int_indexes)==len(self.lines):
                break
            p0 = Point(track[i])
            p1 = Point(track[i+1])
            for i, roi_seg in enumerate(roi_segments):
                if int(self.roi_group[i]) in int_indexes:
                    continue
                q0 , q1 = roi_seg[0], roi_seg[1]
                if doIntersect(p0,p1,q0,q1):
                    int_indexes.append(int(self.roi_group[i]))
                    # compute interseciton point as well
                    if ret_points:
                        int_points.append(self.getIntersectionPoint(p0, p1, q0, q1))

        if ret_points:
            return int_indexes, int_points
        else:
            return int_indexes
    
    def getIntersectionPoint(self, p0, p1, q0, q1):
        segp = sympy.Line(sympy.Point([p0.x, p0.y]), sympy.Point([p1.x, p1.y]))
        segq = sympy.Line(sympy.Point([q0.x, q0.y]), sympy.Point([q1.x, q1.y]))
        point = segp.intersection(segq)
        return [float(point[0][0]), float(point[0][1])]

    @property
    def area(self):
        return abs(float(self.poly.area))

def is_monotonic(traj):
    if len(traj) < 1:
        return False
    orgin = traj[0]
    max_distance = -1
    for p in traj:
        dp = np.linalg.norm(p - orgin)
        if dp < max_distance: return False
        max_distance = dp
    return True

def str_end_to_moi(str, end):
    str_end_moi = {}
    str_end_moi[(1, 4)] = 1
    str_end_moi[(1, 3)] = 2
    str_end_moi[(1, 2)] = 3
    str_end_moi[(2, 1)] = 4
    str_end_moi[(2, 4)] = 5
    str_end_moi[(2, 3)] = 6
    str_end_moi[(3, 2)] = 7
    str_end_moi[(3, 1)] = 8
    str_end_moi[(3, 4)] = 9
    str_end_moi[(4, 3)] = 10
    str_end_moi[(4, 2)] = 11
    str_end_moi[(4, 1)] = 12
    if (str ,end) in str_end_moi:
        return str_end_moi[(str, end)]
    return -1
  
from counting.counting import group_tracks_by_id