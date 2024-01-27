from Libs import *
from Utils import *

def detect(args,*oargs):
  input_file=args.GTJson
  camera_name=os.path.abspath(args.Dataset).split("/")[-1][-3:]
  name_to_num={
    "lc1":0,
    "lc2":1,
    "sc1":2,
    "sc2":3,
    "sc3":4,
    "sc4":5
  }    
  camera_num= name_to_num[camera_name]    
  f= open(input_file ,'r')

  data= json.load(f)
  id_counter=0
  f.close()
  skip=6
  start=args.StartFrame if args.StartFrame is not None else 0
  i=0
  detections=[]
  uuid_to_id={}
  # first populate the uuid. it ensures consistent id across cameras
  for responses in data:
      for response in responses['camera_responses']:
            for gt in response['annotations']:
                if(gt['cuboid_uuid']) not in uuid_to_id:
                    uuid_to_id[gt['cuboid_uuid']]=id_counter
                    id_counter=id_counter+1

  for responses in data:
      for response in responses['camera_responses']:
          # print(response['camera_used'])
          if (response['camera_used']==camera_num):
              # print(len(response['annotations']))
              for gt in response['annotations']:
                  if(gt['cuboid_uuid']) not in uuid_to_id:
                      raise "this should not happen"
                      uuid_to_id[gt['cuboid_uuid']]=id_counter
                      id_counter=id_counter+1
                  uuid=gt['cuboid_uuid']
                  c = 0 if "Pedestrian" in gt['label'] else 7 if 'Truck' in gt['label'] else 5 if 'Buses' in gt['label'] else 2 if 'Small' in gt['label'] else 1 if 'Unpowered' in gt['label'] else 3
                  id=uuid_to_id[gt['cuboid_uuid']]
                  x1=gt['left']
                  x2=x1+gt['width']
                  y1=gt['top']
                  y2=y1+gt['height']
                  detections.append([start+int(skip*i), c, 1.0,x1,y1,x2,y2, uuid, id])
                  # if(start+int(skip*i)-1 >0):
                  #   detections.append([start+int(skip*i)-1, c, 1.0,x1,y1,x2,y2])
                  # detections.append([start+int(skip*i)+1, c, 1.0,x1,y1,x2,y2])
                    
                  # detections.append([start+int(skip*i), id, x1,y1,x2,y2,c])
                  # mot.append([start+int(skip*i), id, x1,y1, gt['width'], gt['height'], 1, c, 1])
              i=i+1
  detections= np.asarray(detections)
  df=pd.DataFrame(detections,columns=['fn','class','score','x1','y1','x2','y2','uuid', "id"])
  df=df.sort_values('fn').reset_index(drop=True)
  # print(df)
  # print(np.unique(df['class']))
  # print(args.DetectionDetectorPath)
  df.to_csv(args.DetectionDetectorPath, header=None, index=None, sep=',')
  return SucLog("copied gt detections from gt.txt to results/detections/**.HW7GT.txt")
def df(args):
  file_path = args.DetectionDetectorPath
  data = {}
  data["fn"], data["class"], data["score"], data["x1"], data["y1"], data["x2"], data["y2"], data["uuid"], data["id"] = [], [], [], [], [], [], [], [], []
  with open(file_path, "r+") as f:
    lines = f.readlines()
    for line in lines:
      splits = line.split(",")
      fn , clss, score, x1, y1, x2, y2, uuid, id = float(splits[0]), float(splits[1]), float(splits[2]), float(splits[3]),\
                                                   float(splits[4]), float(splits[5]), float(splits[6]), str(splits[7]), int(splits[8])
      data["fn"   ].append(fn)
      data["class"].append(clss)
      data["score"].append(score)
      data["x1"   ].append(x1)
      data["y1"   ].append(y1)
      data["x2"   ].append(x2)
      data["y2"   ].append(y2)
      data["uuid" ].append(uuid)
      data["id"   ].append(id)
  return pd.DataFrame.from_dict(data)

def df_txt(df,text_result_path):
  # store a modified version of detection df to the same txt file
  # used in the post processig part of the detection
  # df is in the same format specified in the df function
  with open(text_result_path, "w") as text_file:
    pass

  with open(text_result_path, "w") as text_file:
    for i, row in tqdm(df.iterrows()):
      frame_num, clss, score, x1, y1, x2, y2, uuid, id = row["fn"], row['class'], row["score"], row["x1"], row["y1"], row["x2"], row["y2"], row["uuid"], row["id"]
      text_file.write(f"{frame_num},{clss},{score},{x1},{y1},{x2},{y2},{uuid},{id}\n")