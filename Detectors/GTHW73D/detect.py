from Libs import *
from Utils import *

def detect(args,*oargs):
  
  
    input_file= args.GT3D
    
    camera_name=args.Dataset.split("/")[-2][-3:]
    extrinsic_file= args.EXTRINSICS_PATH
    intrinsic_file= args.INTRINSICS_PATH
    cap = cv2.VideoCapture(args.Video)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH )
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT )
    dims=np.array([[width],[height]])
    zeros=np.zeros((2,1))
    cap.release()
    
    f=open(extrinsic_file,'r')
    extrinsic_data=json.load(f)
    f.close()
    
    extrinsic_mat=(np.array(extrinsic_data['T_'+camera_name+"_localgta"]))
    # extrinsic_mat=np.matmul(extrinsic_mat, extrinsic_mat2)
    f=open(intrinsic_file,'r')
    intrinsic_data=json.load(f)
    f.close()
    intrinsic_mat=np.array(intrinsic_data[camera_name]['intrinsic_matrix'])
    intrinsic_mat=intrinsic_mat[0:3,0:3]
    # intrinsic_mat= np.zeros((3,4))
    # intrinsic_mat[0:3, 0:3]= i_mat
    f= open(input_file ,'r')
    data= json.load(f)
    f.close()
    skip=6
    start=args.StartFrame if args.StartFrame is not None else 0
    print(start)
    uuid_to_id={}
    id_counter=0
    detections=[]
    i=0
    for responses in data:
        for cuboid in responses['cuboids']:
            if(cuboid['uuid']) not in uuid_to_id:
                uuid_to_id[cuboid['uuid']]=id_counter
                id_counter=id_counter+1
            id=uuid_to_id[cuboid['uuid']]
            c = 0 if "Pedestrian" in cuboid['label'] else 2 if 'Small' in cuboid['label'] else 5 if 'Buses' in cuboid['label'] else 7
            x_pos=cuboid['position']['x']
            y_pos=cuboid['position']['y']
            z_pos=cuboid['position']['z']
            x_dim=cuboid['dimensions']['x'] 
            y_dim=cuboid['dimensions']['y']
            z_dim=cuboid['dimensions']['z'] 
            yaw= cuboid["yaw"]
            roll=cuboid['roll']
            pitch=cuboid['pitch']
            rotation_x=np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw),np.cos(yaw),0], [0, 0, 1]])     
            rotation_z=np.array([[1, 0, 0], [0,np.cos(pitch),-np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])     
            
            hh=y_dim/2
            hw=x_dim/2
            hd=z_dim/2
            center= np.array([x_pos,y_pos,z_pos])
            rotation_y=np.array([[np.cos(roll), 0, -np.sin(roll)], [0,1,0], [np.sin(roll), 0, np.cos(roll)]])   
            rotation=np.dot(np.dot(rotation_z,rotation_x), rotation_y)
            c1= np.array([-hw, -hh,-hd])
            c2= np.array([-hw, hh,-hd])
            c3= np.array([hw, hh,-hd])
            c4= np.array([hw, -hh,-hd])  
            c5= np.array([-hw, -hh,hd])
            c6= np.array([-hw, hh,hd])
            c7= np.array([hw, hh,hd])
            c8= np.array([hw, -hh,hd])            
            c1= rotation@c1
            c2= rotation@c2
            c3= rotation@c3
            c4= rotation@c4
            c5= rotation@c5
            c6= rotation@c6
            c7= rotation@c7
            c8= rotation@c8
            p1=  c1+center
            p2= c2+center
            p3=c3+center
            p4=c4+center
            p5=c5+center
            p6=c6+center
            p7=c7+center
            p8=c8+center      
            # x1,y1,z1= x_pos + (y_dim/2), y_pos + (x_dim/2), z_pos + (z_dim/2)
            # x2,y2,z2 = x1-y_dim, y1, z1
            # x3,y3,z3= x1-y_dim, y1-x_dim, z1
            # x4,y4,z4= x1, y1-x_dim, z1
            # x5,y5,z5= x1, y1, z1-z_dim
            # x6,y6,z6= x1-y_dim, y1, z1-z_dim
            # x7,y7,z7= x1-y_dim, y1-x_dim, z1-z_dim
            # x8,y8,z8= x1, y1-x_dim, z1-z_dim      
            detections.append([start+int(skip*i), c,1,id,
            p1,             
            p2,
            p3,
            p4,
            p5,
            p6,
            p7,
            p8])
        i=i+1
    # detections=np.array(detections)
    # print(detections)
    reprojected_detections=[]
    for detection in detections:
        point1=np.ones((4,1))
        point2=np.ones((4,1))
        point3=np.ones((4,1))
        point4=np.ones((4,1))
        point5=np.ones((4,1))
        point6=np.ones((4,1))
        point7=np.ones((4,1))
        point8=np.ones((4,1))
        p1=detection[4]
        p2=detection[5]
        p3=detection[6]
        p4=detection[7]
        p5=detection[8]
        p6=detection[9]
        p7=detection[10]
        p8=detection[11]
        point1[0:3,0]=p1
        point2[0:3,0]=p2
        point3[0:3,0]=p3
        point4[0:3,0]=p4
        point5[0:3,0]=p5
        point6[0:3,0]=p6
        point7[0:3,0]=p7
        point8[0:3,0]=p8
        
        r1_point1= extrinsic_mat @ point1
        r1_point2=extrinsic_mat @ point2
        r1_point3= extrinsic_mat@ point3
        r1_point4= extrinsic_mat @ point4
        r1_point5=extrinsic_mat @ point5
        r1_point6= extrinsic_mat @ point6
        r1_point7=extrinsic_mat @ point7
        r1_point8=extrinsic_mat@ point8
        
        # r1_point1=extrinsic_mat2.dot(r1_point1)
        # r1_point2=extrinsic_mat2.dot(r1_point2)
        # r1_point3=extrinsic_mat2.dot(r1_point3)
        # r1_point4=extrinsic_mat2.dot(r1_point4)
        # r1_point5=extrinsic_mat2.dot(r1_point5)
        # r1_point6=extrinsic_mat2.dot(r1_point6)
        # r1_point7=extrinsic_mat2.dot(r1_point7)
        # r1_point8=extrinsic_mat2.dot(r1_point8)
        
        r1_point1= r1_point1[0:3]/ r1_point1[3]
        r1_point2= r1_point2[0:3]/ r1_point2[3]
        r1_point3= r1_point3[0:3]/ r1_point3[3]
        r1_point4= r1_point4[0:3]/ r1_point4[3]
        r1_point5= r1_point5[0:3]/ r1_point5[3]
        r1_point6= r1_point6[0:3]/ r1_point6[3]
        r1_point7= r1_point7[0:3]/ r1_point7[3]
        r1_point8= r1_point8[0:3]/ r1_point8[3]
        
        
        
        r2_point1= intrinsic_mat@ r1_point1
        r2_point2= intrinsic_mat@ r1_point2
        r2_point3= intrinsic_mat @ r1_point3
        r2_point4= intrinsic_mat@ r1_point4
        r2_point5= intrinsic_mat @r1_point5
        r2_point6= intrinsic_mat @r1_point6
        r2_point7= intrinsic_mat@ r1_point7
        r2_point8= intrinsic_mat @r1_point8
        
        r2_point1= r2_point1[0:2]/ r2_point1[2]
        r2_point2= r2_point2[0:2]/ r2_point2[2]
        r2_point3= r2_point3[0:2]/ r2_point3[2]
        r2_point4= r2_point4[0:2]/ r2_point4[2]
        r2_point5= r2_point5[0:2]/ r2_point5[2]
        r2_point6= r2_point6[0:2]/ r2_point6[2]
        r2_point7= r2_point7[0:2]/ r2_point7[2]
        r2_point8= r2_point8[0:2]/ r2_point8[2]
        points_arr=[r2_point1,r2_point2,r2_point3,r2_point4,r2_point5,r2_point6,r2_point7,r2_point8]
        # print(points_arr)
        s=0
        for point in points_arr:
            s=s+ int(np.count_nonzero((point>zeros) & (point<dims)) ==2)
        # s= np.count_nonzero((r2_point1>zeros) & (r2_point1<dims)) + np.count_nonzero((r2_point2>zeros) & (r2_point2<dims)) +  np.count_nonzero((r2_point3>zeros) & (r2_point3<dims)) +  np.count_nonzero((r2_point4>zeros) & (r2_point4<dims)) +  np.count_nonzero((r2_point5>zeros) & (r2_point5<dims)) +  np.count_nonzero((r2_point6>zeros) & (r2_point6<dims)) +  np.count_nonzero((r2_point7>zeros) & (r2_point7<dims)) +  np.count_nonzero((r2_point8>zeros) & (r2_point8<dims)) 
        p_arr=[]
        for point in points_arr:
            p_arr.append(point[0][0])
            p_arr.append(point[1][0])
        
        if(s>2):
            reprojected_detections.append([detection[0], detection[1],detection[2],detection[3], *p_arr])
        # reprojected_detections.append([detection[0], detection[1],detection[2], *p_arr])
        
    df=pd.DataFrame(reprojected_detections,columns=['fn','class', 'score', 'id','x1','y1', 'x2','y2' ,'x3','y3' , 'x4','y4' ,'x5','y5', 'x6','y6' ,'x7','y7', 'x8','y8'])
    df=df.sort_values('fn').reset_index(drop=True)
    print(df)
    df.to_csv(args.DetectionDetectorPath, header=None, index=None, sep=',')

    # read 3D GT File 
    # read args.GT3D (you need to add it to Utils see How I created args.GT)
    # read calibration matrices
    # args.INTRINSICS_PATH,  args.EXTRINSICS_PATH  (these pathes are already in Utils)

    # Use matrices and GT3D file to backproject 8 corners of 3D bbox to image
    # each bbox will have the following fields"
    # fn, class , score, id,  x1, y1, ..., x8, y8
    # make sure that the first 4 (x, y) pair belong to the front of bbox
    # write corners in args.DetectionDetectorPath (already in Utils) 

    return SucLog("3D GT files stored undeer Detections/GTHW73D.txt")


def df(args):
  file_path = args.DetectionDetectorPath
  data = {}
  data["fn"], data["class"], data["score"], data["id"] = [], [], [], []
  for i in range(1, 9):
    data[f"x{i}"], data[f"y{i}"] = [], []

  with open(file_path, "r+") as f:
    lines = f.readlines()
    for line in lines:
      splits = line.split(",")
      fn , clss, score, id =  float(splits[0]), float(splits[1]), float(splits[2]), float(splits[3])
      data["fn"].append(fn)
      data["class"].append(clss)
      data["score"].append(score)
      data["id"].append(id)

      for idx in range(4, 4 + 2*8, 2): # 8 points x 2 numbers
        i = int((idx - 2)/2) # number corresponding to point f"x{i}""  : idx = 4, 5 -> i=1
        xi, yi = float(splits[idx]), float(splits[idx+1])
        data[f"x{i}"].append(xi)
        data[f"y{i}"].append(yi)

  return pd.DataFrame.from_dict(data)

def df_txt(df,text_result_path):
  # store a modified version of detection df to the same txt file
  # used in the post processig part of the detection
  # df is in the same format specified in the df function
  with open(text_result_path, "w") as text_file:
    pass

  with open(text_result_path, "w") as text_file:
    for i, row in tqdm(df.iterrows()):
      frame_num, clss, score, id = row["fn"], row['class'], row["score"], row["id"]
      text_file.write(f"{frame_num},{clss},{score},{id}")
      for i in range(1, 9):
        xi, yi = row[f"x{i}"], row[f"y{i}"]
        text_file.write(f",{xi},{yi}")
      text_file.write(f"\n") # for new row