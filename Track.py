# Author: Sajjad P. Savaoji May 4 2022
# This py file will handle all the trackings
from Libs import *
from Utils import *
from Detect import *
from counting.resample_gt_MOI.resample_typical_tracks import track_resample
import Homography
import Maps
import TrackLabeling
import copy
import DSM
from Homography import reproj_point
from counting.counting import group_tracks_by_id

# import all detectros here
# And add their names to the "trackers" dictionary
# -------------------------- 
import Trackers.sort.track
import Trackers.CenterTrack.track
import Trackers.DeepSort.track
import Trackers.ByteTrack.track
import Trackers.gsort.track
import Trackers.OCSort.track
import Trackers.GByteTrack.track
import Trackers.GDeepSort.track
# import Trackers.BOTSort.track
import Trackers.StrongSort.track
import Trackers.GTHW7.track
# --------------------------
trackers = {}
trackers["sort"] = Trackers.sort.track
trackers["CenterTrack"] = Trackers.CenterTrack.track
trackers["DeepSort"] = Trackers.DeepSort.track
trackers["ByteTrack"] = Trackers.ByteTrack.track
trackers["gsort"] = Trackers.gsort.track
trackers["OCSort"] = Trackers.OCSort.track
trackers["GByteTrack"] = Trackers.GByteTrack.track
trackers["GDeepSort"] = Trackers.GDeepSort.track
# trackers["BOTSort"] = Trackers.BOTSort.track
trackers["StrongSort"] = Trackers.StrongSort.track
trackers["GTHW7"] = Trackers.GTHW7.track
# --------------------------

def track(args):
    if args.Tracker not in os.listdir("./Trackers/"):
        return FailLog("Tracker not recognized in ./Trackers/")

    current_tracker = trackers[args.Tracker]
    current_tracker.track(args, detectors)
    # store pkl version of tracked df
    store_df_pickle(args)
    store_df_pickle_backup(args)
    return SucLog("Tracking files stored")

def store_df_pickle(args):
    # should be called after tracking is done and the results are stored in the .txt file
    df = trackers[args.Tracker].df(args)
    df.to_pickle(args.TrackingPkl, protocol=4)

def store_df_pickle_backup(args):
    # should be called after tracking is done and the results are stored in the .txt file
    df = trackers[args.Tracker].df(args)
    df.to_pickle(args.TrackingPklBackUp, protocol=4)
def zoom_at(img, zoom=1, angle=0, coord=None):
    
    cy, cx = [ i/2 for i in img.shape[:-1] ] if coord is None else coord[::-1]
    rot_mat = cv2.getRotationMatrix2D((cx,cy), angle, zoom)
    result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    
    return result
def vistrack(args):
    # current_tracker = trackers[args.Tracker]
    # df = current_tracker.df(args)
    df = pd.read_pickle(args.TrackingPkl)
    if(args.BackprojectionMethod is not None and args.BackprojectionMethod=="Homography"):
        reprojected_df=pd.read_pickle(args.ReprojectedPklMeter)
    elif(args.BackprojectionMethod is not None and args.BackprojectionMethod=="DSM"):
        reprojected_df=pd.read_pickle(args.ReprojectedPkl)
    else:
        reprojected_df=None
        
    video_path = args.Video
    annotated_video_path = args.VisTrackingPth
    # tracks_path = args.TrackingPth

    # if(args.CalcSpeed):
    #     calculate_speeds(args)
    color = (0, 0, 102)
    cap = cv2.VideoCapture(video_path)
    # Check if camera opened successfully
    if (cap.isOpened()== False): return FailLog("could not open input video")

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(frame_height, frame_width)
    out_cap = cv2.VideoWriter(annotated_video_path,cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width,frame_height))
    if(args.CalcDistance):
        out_cap2 = cv2.VideoWriter(args.CalculateDistanceVisPth,cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width,frame_height))        
        print(args.CalculateDistanceVisPth)
    if(args.CalcSpeed):
        out_cap3 = cv2.VideoWriter(args.CalculateSpeedVisPth,cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width,frame_height))        
        print(args.CalculateSpeedVisPth)
    if(args.VisAll):
        out_cap4=cv2.VideoWriter(args.VisAllPth,cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width,frame_height)) 
        frame_limit=50
        id_of_interest=21
        print(args.VisAllPth)    
    if(args.VisClass):
        out_cap5=cv2.VideoWriter(args.VisClassPth,cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width,frame_height))
        print(args.VisClassPth)
    if(args.FindLanes):
        out_cap6=cv2.VideoWriter(args.VisLanesPth,cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width,frame_height))
        print(args.VisLanesPth)
    if(args.FindLanes or args.VisAll):
        reprojected_df,df=find_lanes(args,input_reprojected=reprojected_df, input_normal=df)
    # df=df[df['id']==14]
    # df=df[(df['fn']>=74) & (df['fn']<=85)]
    # reprojected_df=reprojected_df[reprojected_df['id']==14]
    # reprojected_df=reprojected_df[(reprojected_df['fn']>=74) & (reprojected_df['fn']<=85)]
    
    if(args.CalcDistance or args.VisAll):
        reprojected_df,df=find_lanes(args,input_reprojected=reprojected_df, input_normal=df)        
        reprojected_df, df=calculate_distance(args,input_reprojected=reprojected_df,input_normal=df)
    # print(df)
    if(args.CalcSpeed or args.VisAll):
        reprojected_df, df=calculate_speeds(args,input_reprojected=reprojected_df,input_normal=df)
    # print(df['class'])
    color_strs={
        0:["Pedestrian", (0,255,255)],
        1:["Bicycle",(0,255,0)],
        2:["Car ",(255,0,0)],
        5:["Bus",(255,255,255)],
        7:["Truck",(0,0,255)]
    }
    unique_classes=np.unique(df['class'])
    legends=[]
    for c in unique_classes:
        legends.append(color_strs[c])
    print(legends)  
    print(unique_classes)  
    print(args.VisTrackingPth)
    if not args.ForNFrames is None:
        frames = args.ForNFrames
    # Read until video is completed
    zooms=[]    
    for frame_num in tqdm(range(frames)):
        if (not cap.isOpened()):
            break
        # Capture frame-by-frame
        ret, frame = cap.read()
        fr=frame.copy()
        fr2=frame.copy()
        fr3=frame.copy()
        fr5=frame.copy()
        fr6=frame.copy()
        if ret:
            this_frame_tracks = df[df.fn==(frame_num)]
            for i, track in this_frame_tracks.iterrows():
                # plot the bbox + id with colors
                bbid, x1 , y1, x2, y2 = track.id, int(track.x1), int(track.y1), int(track.x2), int(track.y2)
                # print(x1, y1, x2, y2)
                np.random.seed(int(bbid))
                color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'id:', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.putText(frame, f'{int(bbid)}', (x1 + 60, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
                cv2.putText(frame, f'{int(bbid)}', (x1 + 60, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (144, 251, 144), 2)
                # if(args.FindLanes):
                #     if(track.lanelabel!="N/A"):
                #         cv2.putText(frame, f'{track.lanelabel}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                if(args.VisClass):
                    obj_class=track['class']
                    color=class_color_dict[int(obj_class)]
                    cv2.rectangle(fr5, (x1, y1), (x2, y2), color, 2)
                if(args.FindLanes):
                    if(track.intersectionlabel):
                        # pass
                        cv2.rectangle(fr6, (x1, y1), (x2, y2), (0,0,0), 2)
                    elif(track.approachlabel=="N/A" and track.lanelabel=="N/A"):
                        # pass
                        cv2.rectangle(fr6, (x1, y1), (x2, y2), (255,255,255), 2)
                    else:         
                        randseed= int(str(track.approachlabel)+str(track.lanelabel))
                        if(randseed<0):
                            randseed=((np.abs(randseed)) +25)
                        randseed=randseed*8
                        np.random.seed(int(randseed))
                        randcolor = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
                        cv2.rectangle(fr6, (x1, y1), (x2, y2), randcolor, 2)
                        
                                                          
                if(args.CalcDistance):
                    
                    # print(track)
                    distance=track.distance
                    cv2.rectangle(fr, (x1, y1), (x2, y2), (0,0,0), 2)
                    # cv2.putText(fr, f'distance:', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)   
                    if(distance>0):
                        cv2.putText(fr, f'{int(round(distance))}m', (x1 , y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
                    # cv2.putText(fr, f'{bbid}', (x1, y1-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    # cv2.putText(fr, f'{track.fn}', (x1, y1-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    
                    # cv2.putText(fr, f'{int(bbid)}', (x1 + 60, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
                        
                    # cv2.putText(fr, f'id:', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    # cv2.putText(fr, f'{int(bbid)}', (x1 + 60, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
                        
                    stop_lines = args.MetaData["StopLines"]
                if(args.CalcDistance):
                    for line in stop_lines:
                        p1 = tuple(line[0])
                        p2 = tuple(line[1])
                        cv2.line(fr, p1, p2, (128, 128, 0), 25)
                if(args.CalcSpeed):
                    speed=track.speed
                    cv2.rectangle(fr2, (x1, y1), (x2, y2), (0,0,0), 2)
                    cv2.putText(fr2, f'{int(speed)} kmh', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)   
                    # cv2.putText(fr2, f'id:', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    # cv2.putText(fr2, f'{int(bbid)}', (x1 + 60, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
                if(args.VisAll):
                    i=1
                    distance=track.distance
                    speed=track.speed
                    label=track.lanelabel
                    if(track.fn<frame_limit):
                        cv2.rectangle(fr3, (x1, y1), (x2, y2), (0,0,0), 2)
                        if(distance>0):
                            cv2.putText(fr3, f'Distance: {distance:.2f} meters', (x1, y1-100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)                       
                        cv2.putText(fr3, f'Speed: {int(speed)} kmph', (x1, y1-60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)   
                        cv2.putText(fr3, f'Label: {label}', (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)   
                        
                        roi = args.MetaData["StoplineROI"]
                        for i in range(0, len(roi)):
                            p1 = tuple(roi[i-1])
                            p2 = tuple(roi[i])
                            cv2.line(fr3, p1, p2, (128, 128, 0), 15)
                            
                    elif(bbid==id_of_interest):
                        if(track.fn<frame_limit+60):
                            # print(x1,y1,x2,y2)
                            # print(track)
                            coord=np.array([float(x1+((x2-x1)/2)), float(y1+((y2-y1)/2))])
                            cv2.rectangle(fr3, (x1, y1), (x2, y2), (0,0,0), 2)
                            if(distance>0):
                                cv2.putText(fr3, f'Distance: {distance:.2f} meters', (x1, y1-100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)                       
                            cv2.putText(fr3, f'Speed: {int(speed)} kmph', (x1, y1-60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)   
                            cv2.putText(fr3, f'Label: {label}', (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)   
                            # print(coord)
                            # input()
                            zoom=(track.fn-frame_limit)/10
                            zoom= 1 if zoom<1 else zoom
                            zooms.append(zoom)
                            fr3= zoom_at(fr3, zoom=zoom, coord=coord)
                        elif(track.fn>=frame_limit+60 and track.fn<frame_limit+120):
                            coord=np.array([float(x1+((x2-x1)/2)), float(y1+((y2-y1)/2))])
                            cv2.rectangle(fr3, (x1, y1), (x2, y2), (0,0,0), 2)
                            if(distance>0):
                                cv2.putText(fr3, f'Distance: {distance:.2f} meters', (x1, y1-100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)                       
                            cv2.putText(fr3, f'Speed: {int(speed)} kmph', (x1, y1-60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)   
                            cv2.putText(fr3, f'Label: {label}', (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)   
                            # print(coord)
                            # input()
                            fr3= zoom_at(fr3, zoom=60/10, coord=coord)
                            
                            zoomed_frame=fr3
                        elif(track.fn>=frame_limit+120 and track.fn<frame_limit+180):
                            coord=np.array([float(x1+((x2-x1)/2)), float(y1+((y2-y1)/2))])
                            cv2.rectangle(fr3, (x1, y1), (x2, y2), (0,0,0), 2)
                            if(distance>0):
                                cv2.putText(fr3, f'Distance: {distance:.2f} meters', (x1, y1-100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)                       
                            cv2.putText(fr3, f'Speed: {int(speed)} kmph', (x1, y1-60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)   
                            cv2.putText(fr3, f'Label: {label}', (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
                            # roi = args.MetaData["StoplineROI"]
                            # for i in range(0, len(roi)):
                            #     p1 = tuple(roi[i-1])
                            #     p2 = tuple(roi[i])
                            #     cv2.line(fr3, p1, p2, (128, 128, 0), 15)
                            zoom=zooms[-i]
                            i=i+1
                            # zoom= 1 if zoom>1 else zoom
                            fr3= zoom_at(fr3, zoom=zoom, coord=coord)
                        elif(track.fn>=frame_limit+180):
                            cv2.rectangle(fr3, (x1, y1), (x2, y2), (0,0,0), 2)
                            if(distance>0):
                                cv2.putText(fr3, f'Distance: {distance:.2f} meters', (x1, y1-100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)                       
                            cv2.putText(fr3, f'Speed: {int(speed)} kmph', (x1, y1-60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)   
                            cv2.putText(fr3, f'Label: {label}', (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
                    elif(track.fn>=frame_limit+180):
                        roi = args.MetaData["StoplineROI"]
                        for i in range(0, len(roi)):
                            p1 = tuple(roi[i-1])
                            p2 = tuple(roi[i])
                            cv2.line(fr3, p1, p2, (128, 128, 0), 15)

                        
                        
                    # p1=p1s[i]                        
                    # print(p1.x, p1.y)
                    # cv.circle(fr, (int(p1.x),int(p1.y)), radius=2, color=color, thickness=10)
        # alpha = 0.6
        # M = np.load(args.HomographyNPY, allow_pickle=True)[0]
        # roi_rep = []
        # roi = args.MetaData["roi"]
        # roi_group = args.MetaData["roi_group"]
        # for p in roi:
        #     point = np.array([p[0], p[1], 1])
        #     new_point = M.dot(point)
        #     new_point /= new_point[2]
        #     roi_rep.append([int(new_point[0]), int(new_point[1])])
        # img1=frame.copy()
        # rows1, cols1, dim1 = frame.shape
        # poly_path = mplPath.Path(np.array(roi_rep))
        # for i in range(rows1):
        #     for j in range(cols1):
        #         if not poly_path.contains_point([j, i]):
        #             img1[i][j] = [0, 0, 0]    
        # frame = cv.addWeighted(img1, alpha, frame, 1 - alpha, 0)
        out_cap.write(frame)
        if(args.CalcDistance):
            out_cap2.write(fr)    
        if (args.CalcSpeed):
            out_cap3.write(fr2)  
        if(args.VisAll):
            out_cap4.write(fr3)  
        if(args.VisClass):
            for i,leg in enumerate(legends):
                legend, color= leg
                cv2.putText(fr5, f"{legend}", (int(frame_width/2)+75, 100 + (50*(i+1))), cv2.FONT_HERSHEY_SIMPLEX,1.5, color,2)  
            out_cap5.write(fr5)
        if(args.FindLanes):
            VisualizeNumbersAreas=args.MetaData['ApproachCounterArea']
            frame_bboxes = df[df.fn==(frame_num)]                        
                        
            for number in VisualizeNumbersAreas:
                print_place=VisualizeNumbersAreas[number][0]
                # print(number)
                # print(frame_bboxes['combinedlabel'])        
                # print("print_place ",print_place)        
                num_bboxes=np.sum(frame_bboxes['combinedlabel']==str(number))
                cv2.putText(fr6, f"{num_bboxes}", (int(print_place[0]), int(print_place[1])), cv2.FONT_HERSHEY_SIMPLEX,1.5, (180,105,255),5)  
                # print(num_bboxes)
                # print(frame_bboxes[frame_bboxes['combinedlabel']==str(number)])
            out_cap6.write(fr6)
        
    print(df)
    cap.release()
    out_cap.release()
    if(args.CalcDistance):
        out_cap2.release()
    if(args.CalcSpeed):
        out_cap3.release()
    if(args.VisAll):
        out_cap4.release()
    if(args.VisClass):
        out_cap5.release()
        
    return SucLog("track vis successful")
def meter_per_pixel(center, zoom=19):
    m_per_p = 156543.03392 * np.cos(center[0] * np.pi / 180) / np.power(2, zoom)
    return m_per_p
def adaptive_gaussian_filter(speed_values, scale_factor=2.0):
    # Calculate the local standard deviation of the data
    local_std = np.std(speed_values)
    
    # Calculate the adaptive sigma based on the scale factor and local standard deviation
    adaptive_sigma = scale_factor * local_std
    
    # Apply Gaussian filter with the adaptive sigma
    smoothed_speeds = gaussian_filter1d(speed_values, sigma=adaptive_sigma, mode='nearest')
    
    return smoothed_speeds
def adaptive_value_proportional_gaussian_filter(speed_values, scale_factor=2.0):
    # Calculate the local moving average of the data
    moving_average = np.convolve(speed_values, np.ones(3)/3, mode='same')  # Adjust the kernel size as needed
    
    # Calculate the adaptive sigma based on the scale factor and local moving average
    
    smoothed_speeds = np.zeros_like(speed_values)

    # Iterate through each point and apply the filter with a variable sigma
    for i in range(len(speed_values)):
        # Calculate the adaptive sigma based on the scale factor and local moving average
        adaptive_sigma = scale_factor * moving_average[i]

        # Apply Gaussian filter with the adaptive sigma
        smoothed_speeds[i] = gaussian_filter1d(speed_values, sigma=adaptive_sigma, mode='nearest')[i]    
    return smoothed_speeds

def find_lanes(args, input_reprojected=None, input_normal=None):
    # roi_to_label={
    #     "right_turn":"Right Turn",
    #     "straight_1":"Straight",
    #     "straight_2":"Straight",
    #     "left_turn":"Left Turn",
    #     "bus_lane_1":"Bus Lane",
    #     "bus_lane_2":"Bus Lane",
    #     "non_approach":"Leaving Intersection"
    # }
    roi_to_label=[
    #    "Right Turn",
    #     "Straight",
    #    "Straight",
    #     "Left Turn",
    #     "Bus Lane",
    #     "Bus Lane",
    #     "Leaving Intersection"
    ]
    
    # if args.BackprojectionMethod=="Homography":   
    df=pd.read_pickle(args.ReprojectedPkl) if input_reprojected is None else input_reprojected 
    track_df=pd.read_pickle(args.TrackingPkl) if input_normal is None else input_normal
    rois = args.MetaData["LaneRois"]
    label1=[]
    label2=[]
    label3=[]
    poly_paths=[]    
    for label in rois:
        for i, roi_arr in enumerate(rois[label]):
            pattern1 = r'^approaches+'  
            pattern2=r'^non_approaches+'     
            for q,r in enumerate(roi_arr):
                if(re.match(pattern1,label)):
                    label1.append(int(i+1))
                elif(re.match(pattern2,label)):
                    label1.append(int(-1*(i+1)))
                                
                roi=roi_arr[r]
                poly_paths.append(mplPath.Path(np.array(roi)))
                label2.append(int(q+1))
    intersection_roi=args.MetaData["StoplineROI"] if "StoplineROI" in args.MetaData else args["roi"]
    intersection_roi_rep=[]
    homography_path = args.HomographyNPY
    if(args.BackprojectionMethod=="Homograpy"):
        M = np.load(homography_path, allow_pickle=True)[0]
        r = meter_per_pixel(args.MetaData['center'])     
        
    elif(args.BackprojectionMethod=="DSM"):
        orthophoto_win_tif_obj, __ = DSM.load_orthophoto(args.OrthoPhotoTif)
        if not os.path.exists(args.ToGroundRaster):
            coords = DSM.load_dsm_points(args)
            intrinsic_dict = DSM.load_json_file(args.INTRINSICS_PATH)
            projection_dict = DSM.load_json_file(args.EXTRINSICS_PATH)
            DSM.create_raster(args, args.MetaData["camera"], coords, projection_dict, intrinsic_dict)
        
        with open(args.ToGroundRaster, 'rb') as f:
            GroundRaster = DSM.pickle.load(f)
    else:
        raise NotImplementedError(f"Reprojection method {args.BackprojectionMethod} not supported to find lanes")
           
    for p in intersection_roi:
        if(args.BackprojectionMethod=="Homography"):
            new_point = reproj_point(args, p[0], p[1], "Homography", M=M)
            new_point=[(int(r*new_point[0]), int(r*new_point[1]))]
        elif(args.BackprojectionMethod=="DSM"):

            new_point = reproj_point(args, p[0], p[1], "DSM", GroundRaster=GroundRaster, TifObj = orthophoto_win_tif_obj)
        intersection_roi_rep.append((new_point[0], new_point[1]))
    poly_path_roi = mplPath.Path(np.array(intersection_roi_rep))
    
    # for roi in rois:
    #     roi_arr= rois[roi]
    #     poly_paths.append(mplPath.Path(np.array(roi_arr)))
    approach_labels=[]
    lane_labels=[]
    intersection_labels=[]
    combined_labels=[]
    for index, row in tqdm(df.iterrows()):
        x,y=row.xcp, row.ycp
        point=[x,y]
        approach_label="N/A"
        lane_label="N/A"
        combined_label="N/A"
        intersection_label=True
        for i, path in enumerate(poly_paths):
            if(path.contains_point(point)):
                approach_label=label1[i]
                lane_label=label2[i]
                combined_label=str(approach_label)+str(lane_label)
                intersection_label=False  
                break
            if(approach_label=="N/A" and lane_label=="N/A"):
                
                if(poly_path_roi.contains_point(point)):
                    intersection_label=True
                else:
                    intersection_label=False
        approach_labels.append(approach_label)
        lane_labels.append(lane_label)
        intersection_labels.append(intersection_label)
        combined_labels.append(combined_label)
    track_df['approachlabel']=approach_labels
    df['approachlabel']=approach_labels
    
    track_df['intersectionlabel']=intersection_labels
    df['intersectionlabel']=intersection_labels
    track_df['lanelabel']=lane_labels
    df['lanelabel']=lane_labels    
    track_df['combinedlabel']=combined_labels
    df['combinedlabel']=combined_labels
    
    return df,track_df        
    
def calculate_speeds(args, input_reprojected=None, input_normal=None):
    cap = cv2.VideoCapture(args.Video)

    if (cap.isOpened()== False): 
        return FailLog("Error opening video stream or file")
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    if args.BackprojectionMethod=="Homography":    
        print("Calculating Speeds! with homography")
        df = pd.read_pickle(args.ReprojectedPklMeter) if input_reprojected is None else input_reprojected  
        track_df=pd.read_pickle(args.TrackingPkl) if input_normal is None else input_normal
        ids= np.unique(df['id'])
        speeds=np.zeros(len(df))
        for track_id in tqdm(ids):
            
            bbox_idx=np.where(df['id']==track_id)[0]
            displacements=np.zeros(len(bbox_idx))
            
            frame_differences=np.zeros(len(bbox_idx))
            bboxes=df.iloc[bbox_idx]
            frames=sorted(bboxes['fn'])
            first_frame=bboxes[bboxes['fn']==frames[0]]
            final_frame=bboxes[bboxes['fn']==frames[-1]]
            first_position=np.array([first_frame.x, first_frame.y])
            final_position=np.array([final_frame.x, final_frame.y])
            global_displacement=final_position-first_position
            normalized_global_dispacement=global_displacement/np.linalg.norm(global_displacement)
            for i in range(len(frames)):
                if(i<len(frames)-1):
                    current_frame=bboxes[bboxes['fn']==frames[i]]
                    next_frame=bboxes[bboxes['fn']==frames[i+1]]
                    current_position=np.array([current_frame.x, current_frame.y])
                    next_position=np.array([next_frame.x, next_frame.y])
                    displacement= (next_position-current_position)
                    normalized_displacement=displacement/np.linalg.norm(global_displacement)
                    dot_product = np.dot(normalized_global_dispacement.T, normalized_displacement)                    
                    displacement=np.linalg.norm(displacement)
                    if(dot_product>0):
                        displacement=displacement
                    elif(dot_product<0):
                        displacement=-displacement
                    else:
                        displacement=displacement
                    frame_diff=int(next_frame.fn)-int(current_frame.fn)
                    time_diff=frame_diff/fps
                    # speed=displacement/time_diff
                    displacements[i]=displacement
                    frame_differences[i]=time_diff
                    
                    # speeds[bbox_idx[i]]=speed*3.6
                    
                    # time_difference=np.abs()
                else:
                    displacements[i]=0
                    frame_differences[i]=1
            smoothed_displacements= displacements                    
            # smoothed_displacements=gaussian_filter1d(displacements, sigma=0.5, mode='nearest')
            speeds[bbox_idx]= 3.6*(smoothed_displacements/frame_differences)
        df['speed']=speeds
        track_df['speed']=speeds
        neighbourhood=30
        
        for track_id in tqdm(ids):
            bbox_idx=np.where(df['id']==track_id)[0]
            bboxes=df.iloc[bbox_idx]
            
            s=np.array(bboxes['speed'])*np.array(bboxes['score'])
            if(len(s)>=neighbourhood):
                new_speeds=gaussian_filter1d(s, sigma=neighbourhood, mode='reflect')
                # s = scipy.ndimage.minimum_filter1d(s, size=neighbourhood)
                # degree = 2
                # coefficients = np.polyfit(time_steps, s, degree)
                # polynomial_fit = np.poly1d(coefficients)
                # x_fit = np.linspace(min(time_steps), max(time_steps), len(time_steps))
                # y_fit = polynomial_fit(x_fit)
                # new_speeds=y_fit
            else:
                neighbourhood=len(s)
                new_speeds=gaussian_filter1d(s, sigma=neighbourhood, mode='reflect')
                
                new_speeds=s
            # if(len(s)>=5):
            # s[0]=0
            # new_speeds=gaussian_filter1d(s, sigma=5, mode='nearest')
            # if(len(s)==1):
            #     new_speeds=s
            # else:
            #     # new_speeds=adaptive_value_proportional_gaussian_filter(s)
            #     new_speeds=gaussian_filter1d(s, sigma=15, mode='nearest')
            #     # new_speeds=s
            # # else:
            #     # new_speeds=s
            new_speeds[new_speeds<0]=0
            speeds[bbox_idx]=new_speeds
        df['speed']=speeds
        track_df['speed']=speeds
            
        return df, track_df
    else:
        print("Calculating Speeds! with DSM")
        df = pd.read_pickle(args.ReprojectedPkl)  if input_reprojected is None else input_reprojected
        track_df=pd.read_pickle(args.TrackingPkl) if input_normal is None else input_normal
        ids= np.unique(df['id'])
        speeds=np.zeros(len(df))
        orthophoto_win_tif_obj, __ = DSM.load_orthophoto(args.OrthoPhotoTif)
        if not os.path.exists(args.ToGroundRaster):
            coords = DSM.load_dsm_points(args)
            intrinsic_dict = DSM.load_json_file(args.INTRINSICS_PATH)
            projection_dict = DSM.load_json_file(args.EXTRINSICS_PATH)
            DSM.create_raster(args, args.MetaData["camera"], coords, projection_dict, intrinsic_dict)
        
        with open(args.ToGroundRaster, 'rb') as f:
            GroundRaster = DSM.pickle.load(f)      
        spatial_resolution=np.array(orthophoto_win_tif_obj.res)
              
        for track_id in tqdm(ids):
            
            bbox_idx=np.where(df['id']==track_id)[0]
            displacements=np.zeros(len(bbox_idx))
            
            frame_differences=np.zeros(len(bbox_idx))
            bboxes=df.iloc[bbox_idx]
            frames=sorted(bboxes['fn'])
            first_frame=bboxes[bboxes['fn']==frames[0]]
            final_frame=bboxes[bboxes['fn']==frames[-1]]
            first_position=np.array([first_frame.x, first_frame.y])
            final_position=np.array([final_frame.x, final_frame.y])
            global_displacement=spatial_resolution[0]*(final_position-first_position)
            normalized_global_dispacement=global_displacement/np.linalg.norm(global_displacement)
            for i in range(len(frames)):
                if(i<len(frames)-1):
                    current_frame=bboxes[bboxes['fn']==frames[i]]
                    next_frame=bboxes[bboxes['fn']==frames[i+1]]
                    current_position=np.array([current_frame.x, current_frame.y])
                    next_position=np.array([next_frame.x, next_frame.y])
                    displacement= spatial_resolution[0]*(next_position-current_position)
                    normalized_displacement=displacement/np.linalg.norm(global_displacement)
                    dot_product = np.dot(normalized_global_dispacement.T, normalized_displacement)                    
                    displacement=np.linalg.norm(displacement)
                    if(dot_product>0):
                        displacement=displacement
                    elif(dot_product<0):
                        displacement=-displacement
                    else:
                        displacement=displacement
                    frame_diff=int(next_frame.fn)-int(current_frame.fn)
                    time_diff=frame_diff/fps
                    # speed=displacement/time_diff
                    displacements[i]=displacement
                    frame_differences[i]=time_diff
                else:
                    displacements[i]=0
                    frame_differences[i]=1
            # smoothed_displacements= displacements                    
            smoothed_displacements=gaussian_filter1d(displacements, sigma=5, mode='reflect')
            speeds[bbox_idx]= 3.6*(smoothed_displacements/frame_differences)
        df['speed']=speeds
        track_df['speed']=speeds
        neighbourhood=30
        
        for track_id in tqdm(ids):
            bbox_idx=np.where(df['id']==track_id)[0]
            bboxes=df.iloc[bbox_idx]
            
            s=np.array(bboxes['speed'])*np.array(bboxes['score'])
            if(len(s)>=neighbourhood):
                # s = scipy.ndimage.minimum_filter1d(s, size=3)
                
                new_speeds=gaussian_filter1d(s, sigma=neighbourhood, mode='reflect')
                # s = scipy.ndimage.minimum_filter1d(s, size=neighbourhood)
                # degree = 2
                # coefficients = np.polyfit(time_steps, s, degree)
                # polynomial_fit = np.poly1d(coefficients)
                # x_fit = np.linspace(min(time_steps), max(time_steps), len(time_steps))
                # y_fit = polynomial_fit(x_fit)
                # new_speeds=y_fit
            else:
                neighbourhood=len(s)
                new_speeds=gaussian_filter1d(s, sigma=neighbourhood, mode='reflect')
                
                new_speeds=s
            # if(len(s)>=5):
            # s[0]=0
            # new_speeds=gaussian_filter1d(s, sigma=5, mode='nearest')
            # if(len(s)==1):
            #     new_speeds=s
            # else:
            #     # new_speeds=adaptive_value_proportional_gaussian_filter(s)
            #     new_speeds=gaussian_filter1d(s, sigma=15, mode='nearest')
            #     # new_speeds=s
            # # else:
            #     # new_speeds=s
            new_speeds[new_speeds<0]=0
            speeds[bbox_idx]=new_speeds
        df['speed']=speeds
        track_df['speed']=speeds
            
        return df, track_df
    

def calculate_distance(args, input_reprojected=None, input_normal=None):
    print(f"Calculating distances with {args.BackprojectionMethod}")
    print(args.ReprojectedPkl)
    if args.BackprojectionMethod=="Homography":
        df = pd.read_pickle(args.ReprojectedPklMeter) if input_reprojected is None else input_reprojected
        track_df=pd.read_pickle(args.TrackingPkl)if input_normal is None else input_normal 
        video_path = args.Video
        cap = cv2.VideoCapture(video_path)
        # Check if camera opened successfully
        if (cap.isOpened()== False): return FailLog("could not open input video")

        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        M = np.load(args.HomographyNPY, allow_pickle=True)[0]
        roi_rep = []
        # roi = args.MetaData["StoplineROI"]
        stop_lines=args.MetaData['StopLines']
        stop_lines_rep=[]
        r = meter_per_pixel(args.MetaData['center'])        
        for line in stop_lines:
            stop_line_rep=[]
            for p in line:
                point = np.array([p[0], p[1], 1])
                new_point = M.dot(point)
                new_point /= new_point[2]
                stop_line_rep.append((int(r*new_point[0]), int(r*new_point[1])))
            stop_lines_rep.append(LineString(np.array(stop_line_rep)))
            # point = np.array([p[0], p[1], 1])
            # new_point = M.dot(point)
            # new_point /= new_point[2]
            # roi_rep.append((int(r*new_point[0]), int(r*new_point[1])))
        # poly_path = mplPath.Path(np.array(roi_rep))
        # poly=Polygon(roi_rep)
        distances=[]
        p1s=[]
        for frame_num in tqdm(range(frames)):
            this_frame_tracks = df[df.fn==(frame_num)]
            for i, track in this_frame_tracks.iterrows():
                # plot the bbox + id with colors
                # bbid, x , y = track.id, int(track.x), int(track.y)
                pcp=[track.x, track.y]
                if track.approachlabel!="N/A" and track.approachlabel>0 and not(track.intersectionlabel):
                    point1=(track.x1, track.y2)
                    point2=(track.x2, track.y2)
                    point1_reproj=reproj_point(args, point1[0], point1[1], "Homography", M=M)
                    point1_reproj=((r*point1_reproj[0]), (r*point1_reproj[1]))
                    point2_reproj=reproj_point(args, point2[0], point2[1], "Homography", M=M)
                    point2_reproj=((r*point2_reproj[0]), (r*point2_reproj[1]))
                    
                    point1S=SPoint(point1_reproj[0], point1_reproj[1])
                    point2S=SPoint(point2_reproj[0], point2_reproj[1])
                    
                    pcp=SPoint(track.x, track.y)   
                    p1, p2 = nearest_points(stop_lines_rep[track.approachlabel-1], pcp)
                    point1p1, p2 = nearest_points(stop_lines_rep[track.approachlabel-1], point1S)
                    point2p1, p2 = nearest_points(stop_lines_rep[track.approachlabel-1], point2S)
                    distance1=pcp.distance(p1)
                    distance2= point1S.distance(point1p1)
                    distance3=point2S.distance(point2p1)
                    # print(frame_num)
                    # print(track.id)
                    # print(point1, point1S)
                    # print(point2, point2S)
                    # print((track.xcp, track.ycp), pcp)
                    # print(distance1, distance2, distance3)
                    # input()
                    distances.append(min(distance1,distance2,distance3))
                    p1s.append(p1)
                else:
                    distances.append(-1)
                    p1s.append(SPoint(0,0))
                    
        track_df['distance']=distances
        print(track_df[np.abs(track_df['distance'])<1 ])
        df['distance']=distances
    else:
        df = pd.read_pickle(args.ReprojectedPkl) if input_reprojected is None else input_reprojected
        track_df=pd.read_pickle(args.TrackingPkl) if input_normal is None else input_normal
        video_path = args.Video
        cap = cv2.VideoCapture(video_path)
        # Check if camera opened successfully
        if (cap.isOpened()== False): return FailLog("could not open input video")

        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        roi = args.MetaData["StoplineROI"]
        orthophoto_win_tif_obj, __ = DSM.load_orthophoto(args.OrthoPhotoTif)
        if not os.path.exists(args.ToGroundRaster):
            coords = DSM.load_dsm_points(args)
            intrinsic_dict = DSM.load_json_file(args.INTRINSICS_PATH)
            projection_dict = DSM.load_json_file(args.EXTRINSICS_PATH)
            DSM.create_raster(args, args.MetaData["camera"], coords, projection_dict, intrinsic_dict)
        
        with open(args.ToGroundRaster, 'rb') as f:
            GroundRaster = DSM.pickle.load(f)
        roi_rep=[]
        stop_lines_rep=[]
        stop_lines=args.MetaData['StopLines']
        
        for line in stop_lines:
            stop_line_rep=[]
            for p in line:
                point = np.array([p[0], p[1], 1])
                new_point =reproj_point(args, p[0], p[1], "DSM", GroundRaster=GroundRaster, TifObj = orthophoto_win_tif_obj)
                # new_point /= new_point[2]
                stop_line_rep.append((new_point[0], new_point[1]))
            stop_lines_rep.append(LineString(np.array(stop_line_rep)))
        
        for p in roi:
            new_point = reproj_point(args, p[0], p[1], "DSM", GroundRaster=GroundRaster, TifObj = orthophoto_win_tif_obj)
            roi_rep.append((new_point[0], new_point[1]))
        poly_path = mplPath.Path(np.array(roi_rep))
        poly=Polygon(roi_rep)
        distances=[]
        p1s=[]
        spatial_resulotion=np.array(orthophoto_win_tif_obj.res)
        for frame_num in tqdm(range(frames)):
            this_frame_tracks = df[df.fn==(frame_num)]
            for i, track in this_frame_tracks.iterrows():
                # plot the bbox + id with colors
                # bbid, x , y = track.id, int(track.x), int(track.y)
                pcp=[track.x, track.y]
                if track.approachlabel!="N/A" and track.approachlabel>0 and not(track.intersectionlabel):
                    point1=(track.x1, track.y2)
                    point2=(track.x2, track.y2)
                    point1_reproj=reproj_point(args, point1[0], point1[1],"DSM", GroundRaster=GroundRaster, TifObj = orthophoto_win_tif_obj)
                    point2_reproj=reproj_point(args, point2[0], point2[1], "DSM", GroundRaster=GroundRaster, TifObj = orthophoto_win_tif_obj)
                    
                    point1S=SPoint(point1_reproj[0], point1_reproj[1])
                    point2S=SPoint(point2_reproj[0], point2_reproj[1])
                    
                    pcp=SPoint(track.x, track.y)   
                    p1, p2 = nearest_points(stop_lines_rep[track.approachlabel-1], pcp)
                    point1p1, p2 = nearest_points(stop_lines_rep[track.approachlabel-1], point1S)
                    point2p1, p2 = nearest_points(stop_lines_rep[track.approachlabel-1], point2S)
                    distance1=pcp.distance(p1)
                    distance2= point1S.distance(point1p1)
                    distance3=point2S.distance(point2p1)
                    # print(frame_num)
                    # print(track.id)
                    # print(point1, point1S)
                    # print(point2, point2S)
                    # print((track.xcp, track.ycp), pcp)
                    # print(distance1, distance2, distance3)
                    # input()
                    min_distance=min(distance1,distance2,distance3)
                    distances.append(spatial_resulotion[0]*min_distance)
                    p1s.append(p1)
                else:
                    distances.append(-1)
                    p1s.append(SPoint(0,0))
                    
        
        # spatial_resulotion=[1,1]
        # for frame_num in tqdm(range(frames)):
        #     this_frame_tracks = df[df.fn==(frame_num)]
        #     for i, track in this_frame_tracks.iterrows():
        #         # plot the bbox + id with colors
        #         # bbid, x , y = track.id, int(track.x), int(track.y)
        #         pcp=[track.x, track.y]
        #         # if not poly_path.contains_point(pcp):
        #         pcp=SPoint(track.x, track.y)            
        #         p1, p2 = nearest_points(poly, pcp)
        #         pixel_distance_x=np.abs(pcp.x-p1.x)
        #         pixel_distance_y=np.abs(pcp.y-p1.y)
        #         distance=np.sqrt(((pixel_distance_x)**2 +(pixel_distance_y)**2 ))
                
        #         # distances.append(pcp.distance(p1))
        #         distances.append(distance)
        #         p1s.append(p1)
        #         # else:
        #         # distances.append(0)
        #         # p1s.append(SPoint(0,0))
                    
        track_df['distance']=distances
        df['distance']=distances
        # uniques = np.unique(track_df['distance'])
        # thresh = uniques[-50]
        # print(track_df['distance'] >= thresh)
        # track_df=track_df[track_df['distance'] >= thresh]      
        # df=df[df['distance'] >= thresh]      
          
    return df, track_df

def vistracktop(args):
    df = pd.read_pickle(args.ReprojectedPkl)
    annotated_video_path = args.VisTrackingTopPth
    video_path = args.Video
    cap = cv2.VideoCapture(video_path)
    # Check if camera opened successfully
    print(args.HomographyTopView)
    if (cap.isOpened()== False): return FailLog("could not open input video")

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if(args.CalcDistance):
        _,df=calculate_distance(args)
    img1 = cv.imread(args.HomographyStreetView)
    frame = cv.imread(args.HomographyTopView)
    rows2, cols2, dim2 = frame.shape

    alpha=0.6
    # df=df[(df['id']==22) | (df['id'] ==129)]
    
    # M = np.load(args.HomographyNPY, allow_pickle=True)[0]
    frame_width , frame_height  = cols2 , rows2
    # img12 = cv.warpPerspective(img1, M, (cols2, rows2))
    # img2 = cv.addWeighted(frame, alpha, img12, 1 - alpha, 0)
    img2= copy.deepcopy(frame)

    color = (0, 0, 102)

    out_cap = cv2.VideoWriter(annotated_video_path,cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width,frame_height))
    print(annotated_video_path)
    roi = args.MetaData["roi"]
    orthophoto_win_tif_obj, __ = DSM.load_orthophoto(args.OrthoPhotoTif)
    if not os.path.exists(args.ToGroundRaster):
        coords = DSM.load_dsm_points(args)
        intrinsic_dict = DSM.load_json_file(args.INTRINSICS_PATH)
        projection_dict = DSM.load_json_file(args.EXTRINSICS_PATH)
        DSM.create_raster(args, args.MetaData["camera"], coords, projection_dict, intrinsic_dict)
    print(args.ToGroundRaster)
    with open(args.ToGroundRaster, 'rb') as f:
        GroundRaster = DSM.pickle.load(f)
    roi_rep=[]
    for p in roi:
        new_point = reproj_point(args, p[0], p[1], "DSM", GroundRaster=GroundRaster, TifObj = orthophoto_win_tif_obj)
        roi_rep.append((new_point[0], new_point[1]))

    if not args.ForNFrames is None:
        frames = min(args.ForNFrames, frames)
    # Read until video is completed
    for frame_num in tqdm(range(frames)):
        frame = copy.deepcopy(img2)

        this_frame_tracks = df[df.fn==(frame_num)]
        for i, track in this_frame_tracks.iterrows():
            # plot the bbox + id with colors
            bbid, x , y = track.id, int(track.x), int(track.y)
            # print(x1, y1, x2, y2)
            np.random.seed(int(bbid))
            color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
            cv.circle(frame, (x,y), radius=2, color=color, thickness=3)
            cv2.putText(frame, f'{int(bbid)}', (x + 10, y-2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            cv2.putText(frame, f'{int(bbid)}', (x + 10, y-2), cv2.FONT_HERSHEY_SIMPLEX, 1, (144, 251, 144), 1)
            if(args.CalcDistance):
                distance=track.distance
                # cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # cv2.putText(fr, f'distance:', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)   
                if(distance>0):
                    cv2.putText(frame, f'{distance:.2f}', (x , y-4), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            
            for i in range(0, len(roi_rep)):
                p1 = tuple(roi_rep[i-1])
                p2 = tuple(roi_rep[i])
                cv2.line(frame, p1, p2, (128, 128, 0), 5)
            
        out_cap.write(frame)
    return SucLog("executed vistrack top")

def vistrackmoi(args):
    current_tracker = trackers[args.Tracker]
    df = current_tracker.df(args)
    video_path = args.Video
    annotated_video_path = args.VisTrackingMoIPth
    df_matching = pd.read_csv(args.CountingIdMatchPth)

    dict_matching = {}
    for i, row in df_matching.iterrows():
        dict_matching[int(row['id'])] = int(row['moi'])
        
    cap = cv2.VideoCapture(video_path)
    # Check if camera opened successfully
    if (cap.isOpened()== False): return FailLog("could not open input video")

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    out_cap = cv2.VideoWriter(annotated_video_path,cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width,frame_height))

    # Read until video is completed
    if args.ForNFrames is not None:
        frames = min(frames, args.ForNFrames)
    for frame_num in tqdm(range(frames)):
        if (not cap.isOpened()):
            break
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            this_frame_tracks = df[df.fn==(frame_num+1)]
            for i, track in this_frame_tracks.iterrows():
                # plot the bbox + id with colors
                bbid, x1 , y1, x2, y2 = int(track.id), int(track.x1), int(track.y1), int(track.x2), int(track.y2)
                if bbid in dict_matching:
                    color = moi_color_dict[dict_matching[bbid]]
                else: color = (0, 0, 125)
                # print(x1, y1, x2, y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'id:{bbid}', (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                if bbid in dict_matching:
                    cv2.putText(frame, f'moi:{dict_matching[bbid]}', (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            out_cap.write(frame)

    # When everything done, release the video capture object
    cap.release()
    out_cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()
    return SucLog("Vis Tracking moi file stored")

def trackpostproc(args):
    def update_tracking_changes(df, args):
        print(ProcLog("updating txt, pkl, reprojected, and meter files for tracking"))
        trackers[args.Tracker].df_txt(df, args.TrackingPth)
        store_df_pickle(args)
        Homography.reproject(args, method=args.BackprojectionMethod,
                            source = args.BackprojectSource, from_back_up=False)
        
        Maps.pix2meter(args)

    # restore original tracks in txt and pkl
    print(ProcLog("recover tracking from backup"))
    df = pd.read_pickle(args.TrackingPklBackUp)
    update_tracking_changes(df, args)
    if args.CalcDistance:
        print("Calculating distances and adding to tracking df")
        df,_=calculate_distance(args)
        update_tracking_changes(df,args)
    
    if args.CalcSpeed:
        print("Calculating speeds and adding to the tracking df")
        df,_=calculate_speeds(args)
        update_tracking_changes(df,args)
        

    # apply postprocessing on args.ReprojectedPkLMeter and ReprojectedPkl
    if not args.TrackTh is None:
        df  = remove_short_tracks(args)
        update_tracking_changes(df, args)

    print(f"starting with {len(np.unique(df['class']))} tracks")
    print(np.unique(df['class']))

    # unify classes in each track
    if args.UnifyTrackClass:
        print("unify classes")
        print(f"starting with {len(np.unique(df['id']))} tracks")
        df = unify_classes_in_tracks(df, args)
        print(f"ending with {len(np.unique(df['id']))} tracks")
        update_tracking_changes(df, args)  

    # filter tracks based on classes
    if args.classes_to_keep:
        print("filter classes")
        print(f"starting with {len(np.unique(df['class']))} classes")
        df = filter_det_class(df, args)
        print(f"ending with {len(np.unique(df['class']))} classes")
        update_tracking_changes(df, args)   

    # apply postprocessing on args.TrackingPkl
    if args.MaskROI:
        print("mask ROI")
        print(f"starting with {len(np.unique(df['id']))} tracks")
        df = remove_out_of_ROI(args)
        print(f"ending with {len(np.unique(df['id']))} tracks")
        update_tracking_changes(df, args)

    if args.MaskGPFrame:
        print("mask GP Frame")
        print(f"starting with {len(np.unique(df['id']))} tracks")
        df = remove_out_of_GP_frame(args)
        print(f"ending with {len(np.unique(df['id']))} tracks")
        update_tracking_changes(df, args)

    if args.RemoveInvalidTracks:
        print("removing invalid tracks")
        print(f"starting with {len(np.unique(df['id']))} tracks")
        df = remove_invalid_tracks(args)
        print(f"ending with {len(np.unique(df['id']))} tracks")
        update_tracking_changes(df, args)
        
    if args.SelectDifEdgeInROI:
        print("remove tracks that begin and end in the same roi region")
        print(f"starting with {len(np.unique(df['id']))} tracks")
        df = select_based_on_roi(args, different_roi_edge)
        print(f"ending with {len(np.unique(df['id']))} tracks")
        update_tracking_changes(df, args)
        

    # if args.SelectEndingInROI:
    #     print("select ending in ROI")
    #     print(f"starting with {len(np.unique(df['id']))} tracks")
    #     df = select_based_on_roi(args, end_in_roi)
    #     print(f"ending with {len(np.unique(df['id']))} tracks")
    #     update_tracking_changes(df, args)

    # if args.SelectBeginInROI:
    #     print("select begin in ROI")
    #     print(f"starting with {len(np.unique(df['id']))} tracks")
    #     df = select_based_on_roi(args, begin_in_roi)
    #     print(f"ending with {len(np.unique(df['id']))} tracks")
    #     update_tracking_changes(df, args)

    if args.HasPointsInROI:
        print("select tracks that have points inside ROI")
        print(f"starting with {len(np.unique(df['id']))} tracks")
        df = select_based_on_roi(args, has_points_in_roi)
        print(f"ending with {len(np.unique(df['id']))} tracks")
        update_tracking_changes(df, args)

    if args.MovesInROI:
        print("select tracks that move at least args.resampleTH in ROI")
        print(f"starting with {len(np.unique(df['id']))} tracks")
        df = select_based_on_roi(args, moves_in_roi)
        print(f"ending with {len(np.unique(df['id']))} tracks")
        update_tracking_changes(df, args)

    if args.CrossROI:
        print("select tracks that cross roi at least once")
        print(f"starting with {len(np.unique(df['id']))} tracks")
        df = select_based_on_roi(args, cross_roi, resample_tracks=True)
        print(f"ending with {len(np.unique(df['id']))} tracks")
        update_tracking_changes(df, args)

    if args.CrossROIMulti:
        print("select tracks that cross roi multiple edges")
        print(f"starting with {len(np.unique(df['id']))} tracks")
        df = select_based_on_roi(args, cross_roi_multiple, resample_tracks=True)
        print(f"ending with {len(np.unique(df['id']))} tracks")
        update_tracking_changes(df, args)

    if args.JustEnterROI:
        print("select tracks that just enter roi")
        print(f"starting with {len(np.unique(df['id']))} tracks")
        df = select_based_on_roi(args, just_enter_roi, resample_tracks=True)
        print(f"ending with {len(np.unique(df['id']))} tracks")
        update_tracking_changes(df, args)

    if args.JustExitROI:
        print("select tracks that just exit roi")
        print(f"starting with {len(np.unique(df['id']))} tracks")
        df = select_based_on_roi(args, just_exit_roi, resample_tracks=True)
        print(f"ending with {len(np.unique(df['id']))} tracks")
        update_tracking_changes(df, args)

    if args.WithinROI:
        print("select tracks that are completely within roi")
        print(f"starting with {len(np.unique(df['id']))} tracks")
        df = select_based_on_roi(args, within_roi, resample_tracks=True)
        print(f"ending with {len(np.unique(df['id']))} tracks")
        update_tracking_changes(df, args)

    if args.ExitOrCrossROI:
        print("select tracks that either exit roi or cross roi multi")
        print(f"starting with {len(np.unique(df['id']))} tracks")
        df = select_based_on_roi(args, just_exit_or_cross_multi, resample_tracks=True)
        print(f"ending with {len(np.unique(df['id']))} tracks")
        update_tracking_changes(df, args)

    if args.EnterOrCrossROI:
        print("select tracks that either enter roi or cross roi multi")
        print(f"starting with {len(np.unique(df['id']))} tracks")
        df = select_based_on_roi(args, just_enter_or_cross_multi, resample_tracks=True)
        print(f"ending with {len(np.unique(df['id']))} tracks")
        update_tracking_changes(df, args)
    
    if args.SelectToBeCounted:
        print("select tracks to be counted")
        print(f"starting with {len(np.unique(df['id']))} tracks")
        df = select_based_on_roi(args, to_be_counted, resample_tracks=True)
        print(f"ending with {len(np.unique(df['id']))} tracks")
        update_tracking_changes(df, args)

    if args.Interpolate:
        print("Interpolate Tracks")
        print(f"starting with {len(np.unique(df['id']))} tracks")
        df = interpolate_tracks(args)
        print(f"ending with {len(np.unique(df['id']))} tracks")
        update_tracking_changes(df, args)
    
    if args.MaskGT:
        print("Masking GT")
        print(f"starting with {len(np.unique(df['id']))} tracks")
        df = mask_gt(df,args)
        print(f"ending with {len(np.unique(df['id']))} tracks")
        update_tracking_changes(df, args)
        

    return SucLog("track post processing executed with no error")

def mask_gt(df,args):
    gt_df= df
    # cap = cv2.VideoCapture(args.Video)

    # if (cap.isOpened()== False): 
    #     return FailLog("Error opening video stream or file")
    # ret, frame= cap.read()
    m= np.load(args.GTMask)
    mask=[]
    for k in m:
        mask= m[k]
        
    mask=mask.astype(np.uint8)
    mask = 255*((mask - np.min(mask)) / (np.max(mask) - np.min(mask)))
    bbox=gt_df[['x1', 'y1' ,'x2' ,'y2']].to_numpy().astype(np.int64)
    areas= (bbox[:,2] -bbox[:,0]) * (bbox[:,3] -bbox[:,1])
    bbox= bbox.tolist()
    # q=np.array([np.arange(bbox[:,0] , bbox[:,2] ),  np.arange(bbox[:,1], bbox[:,3])])
    non_zero=np.zeros((areas.shape))
    for i,b in enumerate(bbox):
        masked= mask[b[1] : b[3] , b[0]:b[2]]
        # masked_img= frame[b[1] : b[3] , b[0]:b[2]]

        non_zero[i] = np.count_nonzero(masked)
    print(len(gt_df))
    gt_df= gt_df[(non_zero/areas)> 0.7] 
    print(len(gt_df))
    return gt_df

def unify_classes_in_tracks(df, args):
    uids = np.unique(df.id)
    for ui in uids:
        ui_classes = df.loc[df.id == ui, "class"].to_numpy()
        ui_classes = ui_classes[ui_classes >= 0]
        class_major = scipy.stats.mode(ui_classes, keepdims=False)
        class_major = int(class_major[0])
        df.loc[df.id == ui, "class"] = class_major
    return df

# TODO seems like these functions were used for ROI with thresholding
# not the new crossing method

def end_in_roi(pg, traj, th, poly_path, *args, **kwargs):
    p = traj[-1]
    if poly_path.contains_point(p):
            return True
    return False

def begin_in_roi(pg, traj, th, poly_path, *args, **kwargs):
    p = traj[0]
    if poly_path.contains_point(p):
            return True
    return False

def moves_in_roi(pg, traj, th, poly_path, *args, **kwargs):
    in_roi_traj = TrackLabeling.get_in_roi_points(traj, poly_path, return_mask=False)
    if len(in_roi_traj) > 1:
        return True
    return False

def has_points_in_roi(pg, traj, th, poly_path, *args, **kwargs):
    for p in traj:
        if poly_path.contains_point(p):
            return True
    return False

def different_roi_edge(pg, traj, th, poly_path, *args, **kwargs):
    d_end, i_end = pg.distance(traj[-1])
    d_str, i_str = pg.distance(traj[0])
    if d_end<=th and d_str<=th:
        return not i_str == i_end
    return True

def cross_roi(pg, traj, *args, **kwargs):
    int_indxes = pg.doIntersect(traj, ret_points=False)
    if int_indxes:
        return True
    return False

def just_enter_roi(pg, traj, th, poly_path, *args, **kwargs):
    int_indxes = pg.doIntersect(traj, ret_points=False)
    cross_once = len(int_indxes)==1
    start_in_roi = poly_path.contains_point(traj[0])
    end_in_roi = poly_path.contains_point(traj[-1])
    if cross_once and (not start_in_roi) and end_in_roi:
        return True
    else:
        return False
    
def just_exit_roi(pg, traj, th, poly_path, *args, **kwargs):
    int_indxes = pg.doIntersect(traj, ret_points=False)
    cross_once = len(int_indxes)==1
    end_in_roi = poly_path.contains_point(traj[-1])
    start_in_roi = poly_path.contains_point(traj[0])
    if cross_once and (not end_in_roi) and start_in_roi:
        return True
    else:
        return False
    
def within_roi(pg, traj, th, poly_path, *args, **kwargs):
    int_indxes = pg.doIntersect(traj, ret_points=False)
    dont_cross_roi = len(int_indxes)==0
    start_in_roi = poly_path.contains_point(traj[0])
    end_in_roi = poly_path.contains_point(traj[-1])
    if dont_cross_roi and start_in_roi and end_in_roi:
        return True
    else:
        return False
    
def cross_roi_multiple(pg, traj, *args, **kwargs):
    int_indxes = pg.doIntersect(traj, ret_points=False)
    if len(np.unique(int_indxes)) > 1:
        return True
    return False

def just_exit_or_cross_multi(pg, traj, th, poly_path, *args, **kwargs):
    if cross_roi_multiple(pg, traj, th, poly_path) or just_exit_roi(pg, traj, th, poly_path):
        return True
    else:
        return False

def just_enter_or_cross_multi(pg, traj, th, poly_path, *args, **kwargs):
    if cross_roi_multiple(pg, traj, th, poly_path) or just_enter_roi(pg, traj, th, poly_path):
        return True
    else:
        return False
    
def unfinished_track(pg, traj, th, poly_path, *args, **kwargs):
    last_recorded_frame = kwargs["frames"][-1]
    last_video_frame    = kwargs["last_frame"]
    if last_recorded_frame >= last_video_frame - kwargs["UnfinishedTrackFrameTh"]:
        return True
    else:
        return False
    
def unstarted_track(pg, traj, th, poly_path, *args, **kwargs):
    first_recorded_frame  = kwargs["frames"][0]
    first_video_frame    = 0
    if first_recorded_frame <= first_video_frame + kwargs["UnfinishedTrackFrameTh"]:
        return True
    else:
        return False
    
def to_be_counted(pg, traj, th, poly_path, *args, **kwargs):
    if cross_roi_multiple(pg, traj, th, poly_path, *args, **kwargs) or\
       just_enter_roi(pg, traj, th, poly_path, *args, **kwargs):
        return True
    else:
        return False

def interpolate_tracks(args):
    df = pd.read_pickle(args.TrackingPkl)
    columns_to_inter_polate = ["x1", "y1", "x2", "y2", "fn"]
    regular_columns = []
    for col in df.columns:
        if not col in columns_to_inter_polate:
            regular_columns.append(col)

    print(regular_columns)
    print(columns_to_inter_polate)

    intpol_df_data = {}
    for col in df.columns:
        intpol_df_data[col] = []

    unique_ids = np.unique(df["id"].to_numpy())
    for uid in tqdm(unique_ids, desc="IntPolTracks"):
        df_id = df[df["id"] == uid].sort_values(by=["fn"])
        for i in range(1, len(df_id)):
            cur_row = df_id.iloc[i]
            pre_row = df_id.iloc[i-1]
            if cur_row["fn"] == pre_row["fn"]:
                print("two detections with same id in tracking results")
                print(cur_row["fn"])
            assert cur_row["fn"] > pre_row["fn"]
            frame_diff = int(cur_row["fn"] - pre_row["fn"])
            if frame_diff > 1 and frame_diff <= args.InterpolateTh:
                weights = np.linspace(0, 1, frame_diff, endpoint=False)
                for w in weights:
                    for col in regular_columns:
                        intpol_df_data[col].append(pre_row[col])
                    for col in columns_to_inter_polate:
                        pre_value = pre_row[col]
                        cur_value = cur_row[col]
                        intpol_df_data[col].append((1-w)*float(pre_value) + (w)*float(cur_value))
            else:
                for col in df_id.columns:
                    intpol_df_data[col].append(pre_row[col])

        # add the last row as it is   
        for col in df_id.columns:
            intpol_df_data[col].append(df_id.iloc[-1][col])

        interpol_df = pd.DataFrame.from_dict(intpol_df_data).sort_values(by=["fn"])

    print(f"original df len: {len(df)}")
    print(f"interpol df len: {len(interpol_df)}")

    return interpol_df
    
def select_based_on_roi(args, condition, resample_tracks=False):
    # get last frame number in the video
    cap = cv2.VideoCapture(args.Video)
    # Check if camera opened successfully
    if (cap.isOpened()== False): return FailLog("could not open input video")
    last_frame  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    first_frame = int(0)

    df = pd.read_pickle(args.TrackingPkl)

    tracks_path = args.ReprojectedPkl
    tracks_meter_path = args.ReprojectedPklMeter
    meta_data = args.MetaData # dict is already loaded
    HomographyNPY = args.HomographyNPY
    M = np.load(HomographyNPY, allow_pickle=True)[0]

    # load data
    tracks = group_tracks_by_id(pd.read_pickle(tracks_path), gp=True)
    tracks_meter = group_tracks_by_id(pd.read_pickle(tracks_meter_path), gp=True)
    tracks['index_mask'] = tracks_meter['trajectory'].apply(lambda x: track_resample(x, return_mask=True, threshold=args.ResampleTH))

    # create roi polygon on ground plane
    roi_rep = []
    for p in args.MetaData["roi"]:
        point = np.array([p[0], p[1], 1])
        new_point = M.dot(point)
        new_point /= new_point[2]
        roi_rep.append([new_point[0], new_point[1]])

    pg = TrackLabeling.MyPoly(roi_rep, args.MetaData["roi_group"])
    th = args.MetaData["roi_percent"] * np.sqrt(pg.area)
    poly_path = mplPath.Path(np.array(roi_rep))

    ids_to_keep = []
    for i, row in tqdm(tracks.iterrows(), total=len(tracks)):
        if resample_tracks:
            traj = row["trajectory"][row["index_mask"]]
            frames = row.frames[row.index_mask]
        else:
            traj = row["trajectory"]
            frames = row.frames

        if condition(pg, traj, th, poly_path,
                    frames=frames, first_frame=first_frame,
                    last_frame=last_frame, UnfinishedTrackFrameTh=args.UnfinishedTrackFrameTh):
            ids_to_keep.append(row["id"])

    mask = []
    for i, row in df.iterrows():
        if row["id"] in ids_to_keep:
            mask.append(True)
        else:
            mask.append(False)
    return df[mask]


def remove_invalid_tracks(args):
    df = pd.read_pickle(args.TrackingPkl)
    mask = []
    to_remove_ids = []
    ids = np.unique(df["id"])

    for id in ids:
        df_id = df[df["id"] == id]
        if len(df_id) <= 2:
            to_remove_ids.append(id)

    for i, row in df.iterrows():
        if row["id"] in to_remove_ids:
            mask.append(False)
        else:
            mask.append(True)

    return df[mask]

def remove_out_of_GP_frame (args):
    df_reproj = pd.read_pickle(args.ReprojectedPkl)
    df_main   = pd.read_pickle(args.TrackingPkl)
    mask = []

    HomographyNPY = args.HomographyNPY
    M = np.load(HomographyNPY, allow_pickle=True)[0]

    # read top
    frame = cv.imread(args.HomographyTopView)
    rows2, cols2, dim2 = frame.shape
    frame_width , frame_height  = cols2 , rows2

    roi_rep = [[0, 0], [0, frame_height], [frame_width, frame_height], [frame_width, 0]]
    
    poly_path = mplPath.Path(np.array(roi_rep))

    for i, row in tqdm(df_reproj.iterrows(), total=len(df_reproj)):
        x, y = row.x, row.y
        p = [x, y]
        if poly_path.contains_point(p):
        # if pg.encloses_point(p):
            mask.append(True)
        else:
            mask.append(False)
    return df_main[mask]


def remove_out_of_ROI(args):
    df_reproj = pd.read_pickle(args.ReprojectedPkl)
    df_main   = pd.read_pickle(args.TrackingPkl)
    mask = []

    HomographyNPY = args.HomographyNPY
    M = np.load(HomographyNPY, allow_pickle=True)[0]

    roi_rep = []
    for p in args.MetaData["roi"]:
        point = np.array([p[0], p[1], 1])
        new_point = M.dot(point)
        new_point /= new_point[2]
        roi_rep.append([new_point[0], new_point[1]])
    pg = TrackLabeling.MyPoly(roi_rep, args.MetaData["roi_group"])
    poly_path = mplPath.Path(np.array(roi_rep))

    for i, row in tqdm(df_reproj.iterrows(), total=len(df_reproj)):
        x, y = row.x, row.y
        p = [x, y]
        if poly_path.contains_point(p):
        # if pg.encloses_point(p):
            mask.append(True)
        else:
            mask.append(False)
    return df_main[mask]

def remove_short_tracks(args):
    th = args.TrackTh
    df_meter_ungrouped = pd.read_pickle(args.ReprojectedPklMeter)
    df_reg_ungrouped   = pd.read_pickle(args.ReprojectedPkl)
    df_meter = group_tracks_by_id(df_meter_ungrouped, gp=True)
    df_reg   = group_tracks_by_id(df_reg_ungrouped, gp=True)

    main_df = pd.read_pickle(args.TrackingPkl)

    to_remove_ids = []
    # resample tracks
    df_meter['trajectory'] = df_meter['trajectory'].apply(lambda x: track_resample(x))
    # df_reg['trajectory'] = df_reg['trajectory'].apply(lambda x: track_resample(x))
    for i, row in df_meter.iterrows():
        if arc_length(row['trajectory']) < th:
            to_remove_ids.append(row['id'])

    mask = []
    for i, row in main_df.iterrows():
        if row['id'] in to_remove_ids:
            mask.append(False)
        else: mask.append(True)

    return main_df[mask]

def arc_length(track):
        """
        :param track: input track numpy array (M, 2)
        :return: the estimated arc length of the track
        """
        assert track.shape[1] == 2
        accum_dist = 0
        for i in range(1, track.shape[0]):
            dist_ = np.sqrt(np.sum((track[i] - track[i - 1]) ** 2))
            accum_dist += dist_
        return accum_dist