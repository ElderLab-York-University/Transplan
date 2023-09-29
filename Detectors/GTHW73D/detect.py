from Libs import *
from Utils import *

def detect(args,*oargs):
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
  data["fn"], data["class"], data["score"], df["id"] = [], [], [], []
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