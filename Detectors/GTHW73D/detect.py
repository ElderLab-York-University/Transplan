from Libs import *
from Utils import *
import re

def get_columns():
  columns=['fn','class','score','id',
          'x1','y1','x2','y2','x3','y3','x4','y4',
          'x5','y5','x6','y6','x7','y7','x8','y8',
          'x_pos','y_pos','z_pos',
          'x_dim','y_dim','z_dim',
          'yaw','roll',"pitch",
          'uuid', 'label', 'numberOfPoints',
          'distance_to_device','motion_state',
          'traversal_direction', 'camera_used',
          'label_cg', 'label_fg',
          'x2D1', 'y2D1', 'x2D2', 'y2D2']
  return columns

def get_dtypes():
  dtypes =[int,int,float,int,
          float,float,float,float,float,float,float,float,
          float,float,float,float,float,float,float,float,
          float,float,float,
          float,float,float,
          float,float,float,
          str, str, int,
          float,str,
          str, str,
          str, str,
          float, float, float, float]
  return dtypes

def remove_text_in_parentheses(input_string):
    # Define a regular expression pattern to match text between parentheses
    pattern = r"\([^)]*\)"

    # Use re.sub() to replace all matches with an empty string
    result_string = re.sub(pattern, '', input_string)

    return result_string.strip()  # Remove leading and trailing whitespaces

def get_class_from_label(label):
  # need to cahnge this part
  c = 0 if "Pedestrian" in label else 2 if 'Small' in label else 5 if 'Buses' in label else 7
  return c

  mapping = {
    " ": 0
  }
  return mapping[label]

def get_fine_grain_form_label(label):
  return label.split(" - ")[-1]

def get_course_grain_from_label(label):
  return label.split(" - ")[0]

def detect(args,*oargs):
    # load video, inputfile, etc
    input_file= args.GT3D
    camera_name=args.Dataset.split("/")[-1][-3:]
    extrinsic_file= args.EXTRINSICS_PATH
    intrinsic_file= args.INTRINSICS_PATH
    cap = cv2.VideoCapture(args.Video)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH )
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT )
    dims=np.array([[width],[height]])
    cap.release()
    
    # load extrinsics
    with open(extrinsic_file,'r') as f:
      extrinsic_data=json.load(f)
      extrinsic_mat=(np.array(extrinsic_data['T_'+camera_name+"_localgta"]))
    
    # load intrinsics
    with open(intrinsic_file,'r') as f:
      intrinsic_data=json.load(f)
      intrinsic_mat=np.array(intrinsic_data[camera_name]['intrinsic_matrix'])
      intrinsic_mat=intrinsic_mat[0:3,0:3]

    # load annotations
    with open(input_file ,'r') as f:
      data = json.load(f)
    
    # there is gt for every 6 frames
    i=int(0)
    skip=int(6)
    start=int(args.StartFrame) if args.StartFrame is not None else int(0)

    # dictionary for matching str uuid to int track id
    uuid_to_id={}
    id_counter=0

    # list of detections in 3D and 2D
    detections=[]
    reprojected_detections=[]

    for responses in data:
        for cuboid in responses['cuboids']:
            # convert uuid from str to int
            if(cuboid['uuid']) not in uuid_to_id:
                uuid_to_id[cuboid['uuid']]=id_counter
                id_counter=id_counter+1

            id=uuid_to_id[cuboid['uuid']]
            # convert class name to class id
            c = get_class_from_label(remove_text_in_parentheses(cuboid['label']))

            # get dimentions and rotations from data
    
            
            uuid                = cuboid['uuid']
            label               = remove_text_in_parentheses(cuboid['label'])
            label_fg            = get_fine_grain_form_label(label)
            label_cg            = get_course_grain_from_label(label)    
            yaw                 = cuboid["yaw"]
            roll                = cuboid['roll']
            pitch               = cuboid['pitch']
            x_pos               = cuboid['position']['x']
            y_pos               = cuboid['position']['y']
            z_pos               = cuboid['position']['z']
            x_dim               = cuboid['dimensions']['x'] 
            y_dim               = cuboid['dimensions']['y']
            z_dim               = cuboid['dimensions']['z'] 
            numberOfPoints      = cuboid['numberOfPoints']\
                                  if 'numberOfPoints' in cuboid\
                                  else -1
            distance_to_device  = cuboid['distance_to_device']\
                                  if 'distance_to_device' in cuboid\
                                  else -1
            motion_state        = cuboid['attributes']['motion_state']\
                                  if (('attributes' in cuboid) and ('motion_state' in cuboid['attributes']))\
                                  else "UNK"
            traversal_direction = cuboid['attributes']['intersection_traversal_direction']\
                                  if (('attributes' in cuboid) and ('intersection_traversal_direction' in cuboid['attributes']))\
                                  else "UNK"
            camera_used         = cuboid['camera_used']\
                                  if 'camera_used' in cuboid\
                                  else "UNK"
            
            # compute rotations
            rotation_x = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                                   [np.sin(yaw),np.cos(yaw),0], [0, 0, 1]])     
            rotation_z = np.array([[1, 0, 0], [0,np.cos(pitch),-np.sin(pitch)],
                                   [0, np.sin(pitch), np.cos(pitch)]])     
            rotation_y = np.array([[np.cos(roll), 0, -np.sin(roll)], [0,1,0],
                                   [np.sin(roll), 0, np.cos(roll)]])   
            rotation   = np.dot(np.dot(rotation_z,rotation_x), rotation_y)
            
            # compute 8 corners of bbox
            center = np.array([x_pos,y_pos,z_pos])
            hh = y_dim/2
            hw = x_dim/2
            hd = z_dim/2

            # build a template box
            c1 = np.array([-hw, -hh,-hd])
            c2 = np.array([-hw, hh,-hd])
            c3 = np.array([hw, hh,-hd])
            c4 = np.array([hw, -hh,-hd])  
            c5 = np.array([-hw, -hh,hd])
            c6 = np.array([-hw, hh,hd])
            c7 = np.array([hw, hh,hd])
            c8 = np.array([hw, -hh,hd])

            # rotate that template box            
            c1 = rotation @ c1
            c2 = rotation @ c2
            c3 = rotation @ c3
            c4 = rotation @ c4
            c5 = rotation @ c5
            c6 = rotation @ c6
            c7 = rotation @ c7
            c8 = rotation @ c8

            # move template box to correct center
            p1 = c1 + center
            p2 = c2 + center
            p3 = c3 + center
            p4 = c4 + center
            p5 = c5 + center
            p6 = c6 + center
            p7 = c7 + center
            p8 = c8 + center      

            # add corners to ditection list    
            detections.append([ start+ skip*i, c, 1, id,
                              p1,p2,p3,p4,p5,p6,p7,p8,
                              x_pos,y_pos,z_pos,
                              x_dim,y_dim,z_dim,
                              yaw,roll,pitch,
                              uuid, label, numberOfPoints,
                              distance_to_device,motion_state,
                              traversal_direction, camera_used,
                              label_cg, label_fg])
        i=i+1

    # reproject corners from 3D to 2D
    for detection in detections:
        # create empty corner points
        point1 = np.ones((4,1))
        point2 = np.ones((4,1))
        point3 = np.ones((4,1))
        point4 = np.ones((4,1))
        point5 = np.ones((4,1))
        point6 = np.ones((4,1))
        point7 = np.ones((4,1))
        point8 = np.ones((4,1))
        # get corner values from 3D detections
        p1 = detection[4]
        p2 = detection[5]
        p3 = detection[6]
        p4 = detection[7]
        p5 = detection[8]
        p6 = detection[9]
        p7 = detection[10]
        p8 = detection[11]
        # populate corners with values
        point1[0:3,0] = p1
        point2[0:3,0] = p2
        point3[0:3,0] = p3
        point4[0:3,0] = p4
        point5[0:3,0] = p5
        point6[0:3,0] = p6
        point7[0:3,0] = p7
        point8[0:3,0] = p8
        # project to 2D with extrinsic_mat
        r1_point1 = extrinsic_mat @ point1
        r1_point2 = extrinsic_mat @ point2
        r1_point3 = extrinsic_mat @ point3
        r1_point4 = extrinsic_mat @ point4
        r1_point5 = extrinsic_mat @ point5
        r1_point6 = extrinsic_mat @ point6
        r1_point7 = extrinsic_mat @ point7
        r1_point8 = extrinsic_mat @ point8
        # normalize values on last dim
        r1_point1 = r1_point1[0:3] / r1_point1[3]
        r1_point2 = r1_point2[0:3] / r1_point2[3]
        r1_point3 = r1_point3[0:3] / r1_point3[3]
        r1_point4 = r1_point4[0:3] / r1_point4[3]
        r1_point5 = r1_point5[0:3] / r1_point5[3]
        r1_point6 = r1_point6[0:3] / r1_point6[3]
        r1_point7 = r1_point7[0:3] / r1_point7[3]
        r1_point8 = r1_point8[0:3] / r1_point8[3]
        # multiply with intrinsics
        r2_point1 = intrinsic_mat @ r1_point1
        r2_point2 = intrinsic_mat @ r1_point2
        r2_point3 = intrinsic_mat @ r1_point3
        r2_point4 = intrinsic_mat @ r1_point4
        r2_point5 = intrinsic_mat @ r1_point5
        r2_point6 = intrinsic_mat @ r1_point6
        r2_point7 = intrinsic_mat @ r1_point7
        r2_point8 = intrinsic_mat @ r1_point8
        # normalize based on last dim
        r2_point1 = r2_point1[0:2] / r2_point1[2]
        r2_point2 = r2_point2[0:2] / r2_point2[2]
        r2_point3 = r2_point3[0:2] / r2_point3[2]
        r2_point4 = r2_point4[0:2] / r2_point4[2]
        r2_point5 = r2_point5[0:2] / r2_point5[2]
        r2_point6 = r2_point6[0:2] / r2_point6[2]
        r2_point7 = r2_point7[0:2] / r2_point7[2]
        r2_point8 = r2_point8[0:2] / r2_point8[2]
        # put all points in one array
        points_arr=[r2_point1,r2_point2,r2_point3,r2_point4,
                    r2_point5,r2_point6,r2_point7,r2_point8]

        # get enclosing 2D box
        x2D1 = np.min([r2_point1[0],r2_point2[0],r2_point3[0],r2_point4[0],
                      r2_point5[0],r2_point6[0],r2_point7[0],r2_point8[0]])

        x2D2 = np.max([r2_point1[0],r2_point2[0],r2_point3[0],r2_point4[0],
                      r2_point5[0],r2_point6[0],r2_point7[0],r2_point8[0]])

        y2D1 = np.min([r2_point1[1],r2_point2[1],r2_point3[1],r2_point4[1],
                      r2_point5[1],r2_point6[1],r2_point7[1],r2_point8[1]])

        y2D2 = np.max([r2_point1[1],r2_point2[1],r2_point3[1],r2_point4[1],
                      r2_point5[1],r2_point6[1],r2_point7[1],r2_point8[1]])
        
        # check that bbox has atleast 1 corner in frame
        def has_all_corners_in_frame(points_arr, dims):
          def is_in_frame(point, dims):
            if point[0] >= 0       and  point[1] >= 0 and\
               point[0] < dims[0]  and  point[1] < dims[1]:
              return 1
            else:
              return 0

          for point in points_arr:
            if not is_in_frame(point, dims):
              return 0
          return 1   

        def has_1corner_in_frame(points_arr, dims):
          def is_in_frame(point, dims):
            if point[0] >= 0       and  point[1] >= 0 and\
               point[0] < dims[0]  and  point[1] < dims[1]:
              return 1
            else:
              return 0

          num_points_in_frame = 0
          for point in points_arr:
            num_points_in_frame += is_in_frame(point, dims)

          if num_points_in_frame >=1 :
            return 1
          else:
            return 0

        if has_all_corners_in_frame(points_arr, dims):
            p_arr = []
            for point in points_arr:
                p_arr.append(point[0][0])
                p_arr.append(point[1][0])
            reprojected_detections.append([*detection[:4], *p_arr, *detection[12:], x2D1, y2D1, x2D2, y2D2])
        
    df=pd.DataFrame(reprojected_detections,
        columns=get_columns())

    df=df.sort_values('fn').reset_index(drop=True)
    # df.to_csv(args.DetectionDetectorPath, header=None, index=None, sep=',')
    df_txt(df,args.DetectionDetectorPath)

    return SucLog("3D GT files stored under Detections/GTHW73D.txt")

def df(args):
  file_path = args.DetectionDetectorPath
  columns = get_columns()
  dtypes  = get_dtypes()
  data = {}
  for col in columns:
    data[col] = []

  with open(file_path, "r+") as f:
    lines = f.readlines()
    for line in lines:
      splits = line[:-1].split(",")
      assert len(splits) == len(columns)
      for col, dt, value in zip(columns, dtypes, splits):
        data[col].append(dt(value))
  return pd.DataFrame.from_dict(data)

def df_txt(df,text_result_path):
  columns = get_columns()
  # store a modified version of detection df to the same txt file
  # used in the post processig part of the detection
  # df is in the same format specified in the df function
  with open(text_result_path, "w") as text_file:
    pass
  
  with open(text_result_path, "w") as text_file:
    for i, row in tqdm(df.iterrows()):
      line_to_write = ""
      for col in columns:
        line_to_write += f"{row[col]},"
      line_to_write = line_to_write[:-1] + '\n'
      text_file.write(line_to_write)