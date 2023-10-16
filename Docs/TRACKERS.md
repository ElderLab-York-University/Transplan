# Tracking Guide
This pipeline supports:
  * trackers from [mmtracking](https://github.com/open-mmlab/mmtracking/blob/master/README.md?plain=1)
  * custom trackers added by user/source

**NOTE**: before running any functionality of the pipeline makesure to format data according to [our requirements](./).

**NOTE**: there is a full-functionality template in `run.py`.

## Run tracking on video
```
python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --Track
```
Results will be stored under src/Results/Tracking/*.[pkl/txt]

## Visualize tracking
```
python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --VisTrack
```
Results will be stored under src/Results/Visualization/*.mp4


## Tracking post processing
| Flag  | Function|
| ----- | ------- |
|--TrackPostProc | Flag to activate post processing module |
|--TrackTh | Resampling threshold on ground plane |
|--Interpolate | Interpolate tracks|
|--RemoveInvalidTracks | Remove tracks with only one ditection |
|--SelectEndingInROI | Select tracks that end in ROI |
|--SelectBeginInROI | Select tracks that start in ROI |
|--MaskGPFrame | Mask detections that are out of topview frame |
|--HasPointsInROI | Only keep tracks that has points in ROI |
|--MaskROI  | Mask detections that are out of ROI |
|--CrossROI | Keep tracks that cross ROI edge |
|--CrossROIMulti | Keep tracks that cross multiple ROI edges(>=2) |
|--JustEnterROI | Keep tracks that just enter ROI but do not exit |
|--JustExitROI | Keep tracks that exit ROi but do not enter |
|--WithinROI | Keep tracks that neither enter nor exit ROI |

```
python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --TrackPostProc --TrackTh=8 --RemoveInvalidTracks\
--SelectDifEdgeInROI --SelectEndingInROI --SelectBeginInROI --MaskGPFrame --HasPointsInROI --MaskROI  --CrossROI\
--CrossROIMulti --JustEnterROI --JustExitROI --WithinROI --Interpolate --ExitOrCrossROI  --BackprojectSource=tracks\
--TopView=[GoogleMap/OrthoPhoto]
```

## Supported Trackers
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
          <li><a href="configs/mot/deepsort">SORT/DeepSORT (ICIP 2016/2017)</a></li>
          <li><a href="configs/mot/tracktor">Tracktor (ICCV 2019)</a></li>
          <li><a href="configs/mot/qdtrack">QDTrack (CVPR 2021)</a></li>
          <li><a href="configs/mot/bytetrack">ByteTrack (ECCV 2022)</a></li>
          <li><a href="configs/mot/ocsort">OC-SORT (arXiv 2022)</a></li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>


## Add custom trackers
Create a folder with the name of `<TrackerName>` under `./Trackers/`.

Then create two python files `./Trackers/TrackerName/track.py` and `./Trackers/TrackerName/run.py`.

The first file is where our pipeline calls `run.py` with `conda_env=TrackerName` which will get detections and store them as txt under Results forlder.

A template for these two files are provided below.

`track.py`
```
def track(args, *oargs):
    setup(args)
    env_name = args.Tracker
    exec_path = "./Trackers/TrackerName/run.py"
    conda_pyrun(env_name, exec_path, args)
    match_classes(args)

def df(args):
    data = {}
    tracks_path = args.TrackingPth
    tracks = np.loadtxt(tracks_path, delimiter=',')
    # process tracks based on tracker ourput in run.py 
    return pd.DataFrame.from_dict(data)

def df_txt(df, out_path):
    with open(out_path,'w') as out_file:
        for i, row in df.iterrows():
          # write row info to out_file as a line

def match_classes(args):
    '''
    after running tracking this function will add class labels if necessary
    it directly works on txt file and modifies it
    '''
    # make a df from txt file
    data = {}
    tracks_path = args.TrackingPth
    tracks = np.loadtxt(tracks_path, delimiter=',')
    # ...
    track_df = pd.DataFrame.from_dict(data)
    det_df = df = pd.read_pickle(args.DetectionPkl)
    class_labels = []

    frames = np.unique(track_df["fn"])

    for fn in tqdm(frames):
        scores = compute_pairwise_iou(track_df[track_df["fn"] == fn], det_df[det_df["fn"] == fn])
        costs = 1 - scores
        row_indices, col_indices = scipy.optimize.linear_sum_assignment(costs)
        class_labels += list(det_df.iloc[col_indices]["class"])
    
    # add class as a column to df
    track_df["class"] = class_labels

    # write class to txt file
    df_txt(track_df, args.TrackingPth)

def setup(args):
    env_name = args.Tracker
    src_url = "https://github.com/TrackerNameRepo.git"
    rep_path = "./Trackers/TrackerName/TrackerName"
    os.system(f"git clone --recurse-submodules {src_url} {rep_path}")
    make_conda_env(env_name, libs="python=3.8 pip")
    os.system(f"conda run --live-stream -n {args.Tracker} pip install ****")
```

`run.py`
```
if __name__ == "__main__":
    # args in a dictionary here where it was a argparse.NameSpace in the main code
    args = json.loads(sys.argv[-1]) 
    video_path = args["Video"]
    text_result_path = args["DetectionDetectorPath"]
    tracker = TrackerModel(...)
    with open(args["DetectionPkl"],"rb") as f:
            detections=pickle.load(f)
    # pass detections to tracker
    resuults = ...
    with open(args["TrackingPth"],"w") as out_file:
        for row in results:
             # write results for txt file   
```

