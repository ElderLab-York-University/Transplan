# Detection Guide
This pipeline supports:
  * detectors from [mmdetection](https://mmdetection.readthedocs.io/en/latest/)
  * custom detectors added by user/source

**NOTE**: before running any functionality of the pipeline makesure to format data according to [our requirements](./).

**NOTE**: there is a full-functionality template in `run.py`.

## Run detection on video
```
python3 main.py --Dataset={src} --Detector={det} --Detect"
```
Results will be stored under src/Results/Detections/*.[pkl/txt]

## Visualize detections
```
python3 main.py --Dataset={src} --Detector={det} --VisDetect"
```
Results will be stored under src/Results/Visualization/*.mp4


## Detection post processing
* `--DetMask` to maskout detection.
* `--classes_to_kee` to select class object to keep.
```
python3 main.py --Datas`et={src}  --Detector={det} --DetPostProc --DetMask --DetTh=0.50 --classes_to_keep 2 5 7")
```
## Supported detectors
<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Object Detection</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
            <li><a href="configs/cascade_rcnn">Cascade R-CNN (CVPR'2018)</a></li>
            <li><a href="configs/yolo">YOLOv3 (ArXiv'2018)</a></li>
            <li><a href="configs/centernet">CenterNet (CVPR'2019)</a></li>
            <li><a href="configs/detr">DETR (ECCV'2020)</a></li>
            <li><a href="configs/yolox">YOLOX (CVPR'2021)</a></li>
            <li><a href="configs/deformable_detr">Deformable DETR (ICLR'2021)</a></li>
            <li><a href="configs/rtmdet">RTMDet (ArXiv'2022)</a></li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>


## Add custom detectors
Create a folder with the name of `<DetectorName>` under `./Detectors/`.

Then create two python files `./Detectors/DetectorName/detect.py` and `./Detectors/DetectorName/run.py`.

The first file is where our pipeline calls `run.py` with `conda_env=DetectorName` which will get detections and store them as txt under Results forlder.

A template for these two files are provided below.

`detect.py`
```
def detect(args,*oargs):
  setup(args)
  env_name = args.Detector
  exec_path = "./Detectors/<DetectorName>/run.py"
  conda_pyrun(env_name, exec_path, args)

def df(args):
  file_path = args.DetectionDetectorPath
  data = {}
  # read detections from txt file and return as df
  return pd.DataFrame.from_dict(data)

def df_txt(df,text_result_path):
  with open(text_result_path, "w") as text_file:
    for i, row in tqdm(df.iterrows()):
      # write detection details to txt file

def setup(args):
    env_name = args.Detector
    src_url = "CUSTOM_Detector.git"
    rep_path = "./Detectors/<DetectorName>/<DetectorName>"
    os.system(f"git clone {src_url} {rep_path}")
    os.system(f"mkdir ./Detectors/DetectorName/DetectorName/checkpoint_dir")
    os.system(f"wget -c <Check_Point.pt> -O ./Detectors/DetectorName/DetectorName/checkpoint_dir/*.pth")
    if not env_name in get_conda_envs():
        make_conda_env(env_name, libs="python=3.7")
        # install library on conda env
        os.system(f"conda run --live-stream -n {args.Detector} pip install ***")
```

`run.py`
```
if __name__ == "__main__":
  args = json.loads(sys.argv[-1])
  video_path = args["Video"]
  text_result_path = args["DetectionDetectorPath"] 
  model = init_detector(config_file, checkpoint_file, device=device_name)
  video = mmcv.VideoReader(video_path)

  i=0
  with open (text_result_path,"w") as f: 
      for frame in tqdm(video):
          with torch.no_grad():
            result = inference_detector(model, frame)
          res= getbboxes(result)
          # store res to 
          i=i+1
```
