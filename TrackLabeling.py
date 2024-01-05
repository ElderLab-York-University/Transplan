# this file is developed to envoce tracklabeling GUI
from Utils import * 
from Libs import *
from counting.resample_gt_MOI.resample_typical_tracks import track_resample
import Track
import Homography
import DSM

def tracklabelinggui(args):
    export_path = os.path.abspath(args.TrackLabellingExportPth)
    topview = os.path.abspath(args.HomographyTopView)
    clusterspath = os.path.abspath(args.ReprojectedPklCluster)
    cwd = os.getcwd()
    os.chdir("./cluster_labelling_gui/")
    ret = os.system(f"python3 cam_gen.py --Export='{export_path}' --TopView='{topview}' --ClustersPath='{clusterspath}'")
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
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close("all")

    # plot labelled tracks on image domain as well
    alpha = 0.6
    save_path_street = args.VisLabelledTracksPthImage
    tracks = pd.read_pickle(args.TrackLabellingExportPthImage)
    tracks = tracks.sort_values("moi")
    img1 = cv.imread(args.HomographyStreetView)
    M = np.load(args.HomographyNPY, allow_pickle=True)[0]
    rows2, cols2, dim2 = img2.shape

    for i in range(len(tracks)):
        track = tracks.iloc[i]
        traj = track['trajectory']
        moi = track["moi"]
        for j , p in enumerate(traj):
            x , y = int(p[0]), int(p[1])
            c = moi_color_dict[moi]
            try:
                img2 = cv.circle(img1, (x,y), radius=3, color=c, thickness=3)
            except:
                pass

    plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
    plt.axis('off')
    plt.savefig(save_path_street, bbox_inches='tight')
    plt.close("all")

    return SucLog("labeled trackes plotted successfully")

def get_in_roi_points(traj, poly_path, return_mask=False):
    mask = []
    for p in traj:
        if poly_path.contains_point(p):
            mask.append(True)
        else:
            mask.append(False)

    if return_mask: return mask
    else: return traj[mask]

def get_proper_tracks_multi(args_mcs, gp):
    args_flat = flatten_args(args_mcs)
    all_tracks = []
    for args in args_flat:
        tracks = get_proper_tracks(args, gp)
        all_tracks.append(tracks)
    tracks_mcs = pd.concat(all_tracks)
    return tracks_mcs

def get_proper_tracks(args, gp, verbose=True):
    
    tracks_path = args.ReprojectedPkl
    tracks_meter_path = args.ReprojectedPklMeter
    top_image = args.HomographyTopView
    street_image = args.HomographyStreetView
    meta_data = args.MetaData # dict is already loaded

    img = plt.imread(top_image)
    img_street = plt.imread(street_image)
    img1 = cv.imread(top_image if gp else street_image)

    # load data (for both gp and image)
    tracks = group_tracks_by_id(pd.read_pickle(tracks_path), gp=gp, verbose=verbose)
    tracks_meter = group_tracks_by_id(pd.read_pickle(tracks_meter_path), gp=gp, verbose=verbose)
    tracks['index_mask'] = tracks_meter['trajectory'].apply(lambda x: track_resample(x, return_mask=True, threshold=args.ResampleTH))

    # create roi polygon
    method = args.BackprojectionMethod
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

    if gp:
        roi_rep = []
        for p in args.MetaData["roi"]:
            x , y = p[0], p[1]
            new_point = Homography.reproj_point(args, x, y, method,
                        M=M, GroundRaster=GroundRaster, TifObj = orthophoto_win_tif_obj)
            roi_rep.append([new_point[0], new_point[1]])
    else:
        roi_rep = [p for p in args.MetaData["roi"] ]

    pg = MyPoly(roi_rep, args.MetaData["roi_group"])
    th = args.MetaData["roi_percent"] * np.sqrt(pg.area)
    poly_path = mplPath.Path(np.array(roi_rep))

    tracks['ROI_mask'] = tracks.apply(lambda row: get_in_roi_points(row.trajectory[row.index_mask], poly_path, return_mask=True), axis=1)

    # select only tracks that have two interactions with Intersection
    counter = 0
    mask = []
    i_strs = []
    i_ends = []
    p_strs = []
    p_ends = []
    moi = []
    if verbose:
        print("!-!: selecting tacks that have at least two interactions with ROI")
    for i, row in tqdm(tracks.iterrows(), total=len(tracks), disable= not verbose):
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
    if verbose:
        print(f"percentage of complete tracks: {counter/len(tracks)}")
        plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
        plt.savefig(f"multicross_gp={gp}.png")
        plt.close("all")

    # modify dfs with mask
    tracks = tracks[mask]

    # temporarily add info to track dataframe
    tracks['i_str'] = i_strs
    tracks['i_end'] = i_ends
    tracks['moi']   = moi
    tracks['p_str'] = p_strs
    tracks['p_end'] = p_ends

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
        # you should better cluster based on both str and end @TODO
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
        uni_vals, uni_counts = np.unique(tracks_mi.clt, return_counts=True)
        num_tracks_to_get_from_clt_per_moi = min(np.min(uni_counts), 10)
        print(f"nums to take: {num_tracks_to_get_from_clt_per_moi} moi:{mi} num_clt:{len(uni_vals)} tot_tracks_chosen:{len(uni_vals)*num_tracks_to_get_from_clt_per_moi}")
        for uv in uni_vals:
            idxs_mi_clt  = tracks_mi[tracks_mi.clt == uv].index[:num_tracks_to_get_from_clt_per_moi]
            indexes_to_keep += [i for i in idxs_mi_clt]

    return tracks.loc[indexes_to_keep]

def extract_common_tracks_multi(arg_mcs):
    # in our multi setup we will work on ground plane
    # set gp = True
    gp = True
    args_flatten = flatten_args(arg_mcs)
    # all sources should share the same top-view
    top_image = args_flatten[0].HomographyTopView
    # load street image for each source
    street_images = [args.HomographyStreetView for args in args_flatten]
    # we will export comon tracks on all sources
    exportpaths      = [args.TrackLabellingExportPth for args in args_flatten]
    exportpathimages = [args.TrackLabellingExportPthImage for args in args_flatten]
    # tracks of each source
    tracks_paths       = [args.ReprojectedPkl for args in args_flatten]
    tracks_meter_paths = [args.ReprojectedPklMeter for args in args_flatten]
    # load metadata for each source
    meta_datas = [args.MetaData for args in args_flatten]
    # load homographies
    Ms = [np.load(args.HomographyNPY, allow_pickle=True)[0] for args in args_flatten]
    # moi clusters are the same for all sources
    moi_clusters = {}
    for mi in  args_flatten[0].MetaData["moi_clusters"].keys():
        moi_clusters[int(mi)] = args_flatten[0].MetaData["moi_clusters"][mi]
    # load all the images to arrays
    # top view image is the same
    img = plt.imread(top_image)
    img1 = cv.imread(top_image)
    img_top    = plt.imread(top_image)
    # street image is different for each source
    img_streets = [plt.imread(street_image) for street_image in street_images]
    trackss = [get_proper_tracks(args, gp=gp, verbose=False) for args in tqdm(args_flatten)]
    # stack all the dfs on top of eachother
    # resent indexes to avoid duplicate indices
    tracks = pd.concat(trackss, ignore_index=True) 
    #!!! extracted trackss are on ground plane
    # indexes are from merged dfs
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
        plt.imshow(img_top)
        plt.scatter(starts[:, 0], starts[:, 1], c=c_start)
        plt.savefig(f"multi_int_lane_clt_{mi}_gp={gp}.png")
        plt.close("all")

        metric = Metric_Dict["cmm"]
        for c_label in cluster_labels:
            mask_c = c_start == c_label
            tracks_mi_cl = tracks_mi[mask_c]
            indexes_mi_cl = np.array([i for i in range(len(tracks_mi_cl))]).reshape(-1, 1)
            M = np.zeros(shape=(len(indexes_mi_cl), len(indexes_mi_cl)))
            for i in indexes_mi_cl:
                for j in range(int(i)+1, len(indexes_mi_cl)):
                    traj_a = tracks_mi_cl['trajectory'].iloc[int(i)]
                    traj_b = tracks_mi_cl['trajectory'].iloc[int(j)]
                    # only look at the resampled points
                    idx_mask  = tracks_mi_cl['index_mask'].iloc[int(i)]
                    roi_mask  = tracks_mi_cl['ROI_mask'].iloc[int(i)]
                    traj_a    = traj_a[idx_mask][roi_mask]

                    idx_mask = tracks_mi_cl['index_mask'].iloc[int(j)]
                    roi_mask = tracks_mi_cl['ROI_mask'].iloc[int(j)]
                    traj_b  = traj_b[idx_mask][roi_mask]

                    # compute pairwise distance
                    c =  metric(traj_a, traj_b)
                    M[int(i), int(j)] = c
                    M[int(j), int(i)] = c
            M_avg = np.mean(M, axis=0)
            i_c = np.argmin(M_avg)
            # i_c = np.argmax(p_start[mask_c])
            chosen_index = index_mi[mask_c][i_c]
            chosen_indexes.append(chosen_index)

    # save common tracks as labelled tracks on ground plane
    # read all the tracks
    # will reselect proper tracks using df index
    tracks_labelleds = [group_tracks_by_id(pd.read_pickle(tracks_path), gp=gp, verbose=False).loc[tracks_temp.index] for tracks_path, tracks_temp in zip(tracks_paths, trackss)]
    # reset indexes to be consistant
    tracks_labelled = pd.concat(tracks_labelleds, ignore_index=True)
    tracks_labelled = tracks_labelled.loc[chosen_indexes]
    tracks_labelled["moi"] = tracks.loc[chosen_indexes]["moi"].apply(lambda x: int(x))
    for exportpath in exportpaths:
        tracks_labelled.to_pickle(exportpath)

    # tracks will be saved on image plane as well
    # because some of the tracks might not exist in specific image
    # we will projcet ground tracks to image and save them accordingly
    for args in args_flatten:
        M, GroundRaster, orthophoto_win_tif_obj = get_projection_objs(args)
        projected_trajectories = []
        tracks_labelled_this_arg = copy.deepcopy(tracks_labelled)
        for traj in tracks_labelled_this_arg.trajectory:
            projected_traj = []
            for p in traj:
                x, y = p[0], p[1]
                cor = Homography.project_point(args, x, y, args.BackprojectionMethod,
                                M=M, GroundRaster=GroundRaster, TifObj = orthophoto_win_tif_obj)
                cor_x , cor_y = int(cor[0]), int(cor[1])
                projected_traj.append([cor_x, cor_y])
            projected_trajectories.append(projected_traj)
        
        tracks_labelled_this_arg.trajectory = projected_trajectories
        exportpathimage = args.TrackLabellingExportPthImage
        tracks_labelled_this_arg.to_pickle(exportpathimage)

    return SucLog("extracted common tracks based on MSMC")


def extract_common_tracks(args, gp):
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
    img_top    = plt.imread(top_image)

    # get proper tracks
    tracks = get_proper_tracks(args, gp=gp)

    # extract all starts and ends from proper tracks
    starts = []
    ends=[]
    for i, row in tracks.iterrows():
        starts.append(row['p_str'])
        ends.append(row['p_end'])

    starts = np.array(starts)
    ends = np.array(ends)
    plt.imshow(img_top if gp else img_street)
    plt.scatter(starts[:, 0],starts[:, 1], alpha=0.3)
    plt.savefig(f"multicorss_points_gp={gp}.png")
    plt.close("all")

    # cluster tracks and choose common tracks as cluster centers based on CMM
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
        plt.imshow(img_top if gp else img_street)
        plt.scatter(starts[:, 0], starts[:, 1], c=c_start)
        plt.savefig(f"int_lane_clt_{mi}_gp={gp}.png")
        plt.close("all")

        metric = Metric_Dict["cmm"]
        for c_label in cluster_labels:
            mask_c = c_start == c_label
            tracks_mi_cl = tracks_mi[mask_c]
            indexes_mi_cl = np.array([i for i in range(len(tracks_mi_cl))]).reshape(-1, 1)
            M = np.zeros(shape=(len(indexes_mi_cl), len(indexes_mi_cl)))
            for i in indexes_mi_cl:
                for j in range(int(i)+1, len(indexes_mi_cl)):
                    traj_a = tracks_mi_cl['trajectory'].iloc[int(i)]
                    traj_b = tracks_mi_cl['trajectory'].iloc[int(j)]
                    # only look at the resampled points
                    idx_mask  = tracks_mi_cl['index_mask'].iloc[int(i)]
                    roi_mask  = tracks_mi_cl['ROI_mask'].iloc[int(i)]
                    traj_a    = traj_a[idx_mask][roi_mask]

                    idx_mask = tracks_mi_cl['index_mask'].iloc[int(j)]
                    roi_mask = tracks_mi_cl['ROI_mask'].iloc[int(j)]
                    traj_b  = traj_b[idx_mask][roi_mask]

                    # compute pairwise distance
                    c =  metric(traj_a, traj_b)
                    M[int(i), int(j)] = c
                    M[int(j), int(i)] = c
            M_avg = np.mean(M, axis=0)
            i_c = np.argmin(M_avg)
            # i_c = np.argmax(p_start[mask_c])
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

    return SucLog("common tracks extracted successfully")

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

    def all_distances(self, point):
        p = sympy.Point(*point)
        distances = []
        for line, gp in zip(self.lines, self.roi_group):
            d_line = float(line.distance(p))
            if gp <=0 : d_line = float('inf')
            distances.append(d_line)
        distances = np.array(distances)
        min_pos = np.argmin(distances)
        return distances, self.roi_group

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
    
# @TODO change to check for loops
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