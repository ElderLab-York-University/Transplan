from Libs import *
from Utils import *

def track(args, detectors):
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

    # populate uuid_to_id. It ensures consistent id across four camearas
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
                        raise "this should not happen anymore"
                        uuid_to_id[gt['cuboid_uuid']]=id_counter
                        id_counter=id_counter+1
                    c = 0 if "Pedestrian" in gt['label'] else 7 if 'Truck' in gt['label'] else 5 if 'Buses' in gt['label'] else 2 if 'Small' in gt['label'] else 1 if 'Unpowered' in gt['label'] else 3
                    id=uuid_to_id[gt['cuboid_uuid']]
                    uuid= gt['cuboid_uuid']
                    x1=gt['left']
                    x2=x1+gt['width']
                    y1=gt['top']
                    y2=y1+gt['height']
                    # detections.append([start+int(skip*i), c, 1.0,x1,y1,x2,y2])
                    detections.append([start+int(skip*i), id, x1,y1,x2,y2, c, uuid])
                    # mot.append([start+int(skip*i), id, x1,y1, gt['width'], gt['height'], 1, c, 1])
                i=i+1
    detections= np.asarray(detections)
    df=pd.DataFrame(detections,columns=['fn','id','x1','y1','x2','y2','class','uuid'])
    df=df.sort_values('fn').reset_index(drop=True)
    df.to_csv(args.TrackingPth, header=None, index=None, sep=',')

def df(args):
    data = {}
    file_path= args.TrackingPth
    data["fn"], data["id"],  data["x1"], data["y1"], data["x2"], data["y2"], data["class"], data["uuid"] = [], [], [], [], [], [], [], []

    # tracks = np.loadtxt(tracks_path, delimiter=',')
    # data["fn"]    = tracks[:, 0]
    # data["id"]    = tracks[:, 1]
    # data["x1"]    = tracks[:, 2]
    # data["y1"]    = tracks[:, 3]
    # data["x2"]    = tracks[:, 4]
    # data["y2"]    = tracks[:, 5]
    # data['uuid']  = tracks[:, 6]
    # data["class"] = tracks[:, 7]
    with open(file_path, "r+") as f:
        lines = f.readlines()
        for line in lines:
            splits = line.split(",")
            fn , id, x1, y1, x2, y2, clss, uuid = float(splits[0]), float(splits[1]), float(splits[2]), float(splits[3]),\
                                                        float(splits[4]), float(splits[5]), int(splits[6]), str(splits[7])
            data["fn"   ].append(fn)
            data["id"].append(id)
            data["x1"   ].append(x1)
            data["y1"   ].append(y1)
            data["x2"   ].append(x2)
            data["y2"   ].append(y2)
            data["uuid" ].append(uuid)
            data["class"   ].append(clss)

    return pd.DataFrame.from_dict(data)

def df_txt(df, out_path):
    with open(out_path,'w') as out_file:
        for i, row in df.iterrows():
            fn, idd, x1, y1, x2, y2, clss = row['fn'], row['id'], row['x1'], row['y1'], row['x2'], row['y2'], row["class"]
            print('%d,%d,%.4f,%.4f,%.4f,%.4f,%d'%(fn, idd, x1, y1, x2, y2, clss),file=out_file)
            # fn, idd, x1, y1, x2, y2 = row['fn'], row['id'], row['x1'], row['y1'], row['x2'], row['y2']
            # print('%d,%d,%.4f,%.4f,%.4f,%.4f'%(fn, idd, x1, y1, x2, y2),file=out_file)
    # df = pd.read_pickle(args.TrackingPkl)
    # out_path = args.TrackingPth 