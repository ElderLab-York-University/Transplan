# User Guide
For a typical intersection environment the following steps should be taken.

These steps are consistent with `demo.py`.

## Step 0: Dataset structure and required information
In general, the dataset you want to run the pipeline on should have the following format.

```
# Dataset structure should be similar to below
# Data---|
#        |--Split1---|
#        |           |--Segment1---|
#        |           |             |--Source1--|
#        |           |             |           |--VideoName.ext
#________|___________|_____________|___________|--VideoName.metadata.json
```
For each `VideoName.mp4/ext` the **MUST** be a VideoName.meadata.json file.
Consistent with the name of of file it includes metadata related to that intersection.
An example is given below.

`VideoName.metadata.json`
```
{
  "center": [
    43.844412, -79.382311
  ],
  "Address": "HW7 and Leslie",
  "camera": "Source1",
  "roi": [
    [4132, 1057],
    [2786, 834],
    [874, 1020],
    [1226, 2412]     
    ],
  "roi_group":[1, 2, 3, 4],
  "gt":{"1":0, "2":0, "3":0, "4":0, "5":19, "6":1, "7":0, "8":0, "9":0, "10":0, "11":9, "12":1},
  "roi_percent": 0.05,
  "moi_clusters":{"1":3, "2":3, "3":3, "4":3, "5":3, "6":3, "7":3, "8":3, "9":3, "10":3, "11":3, "12":3}
}
```

| key | description |
| --- | ----------- |
| `center` | long lat coordinates of the center of intersecion |
| `adress` | adress of intersection (optional)|
| `camera` | video source in multi-camera environments |
| `roi`    | region of interest containing the intersection formed by connecting the list of coordinates|
| `roi_group` | for flexible grouping, each edge of roi can be taged with a group number(for intersections 4 groups)|
| `gt` | the turn-counts grand truth |
| `moi_clusters` | number of lanes per direction |

## Step 1: Verify structure of your data / Extract Images
To do so, you can extract images from the video.
```
python3 main.py --Dataset={src} --ExtractImages
```
Once you run this command, the will be a folder made called `Results` right next to the video.
You can find your extracted images under `Results/Images`.

## Step 2: Solve homographies
For most of our subtasks we will need a homography planer translation between the camera view and top view.
There is a GUI that you can run to solve this homography.
For the homography you need to have a top view frame `VideoName.homography.top.<TopView>.png` and a camera view frame `VideoName.homography.street.png`. 
If these files are not provided the pipeline will extract camera view from video using `--Frame` flag while top view is extracted using GoogleMaps API using `metadata.json->center`.
```
python3 main.py --Dataset={src}  --HomographyGUI --VisHomographyGUI --Frame=1 --TopView=[GoogleMap/OrthoPhoto]
```
| Flag | Function |
| ---- | -------- |
| `--HomographyGUI` | launch GUI to solve homographies |
| `--VisHomographyGUI` | Visualize homography bu backprojecting cameara view to top view |
| `--Frame` | frame to extract from video as camera view |
| `--TopView` | GoogleMap or OrthoPhoto |

## Step 3: Visualize ROI
```
python3 main.py --Dataset={src} --VisROI --TopView=GoogleMap
```

## Step 4: Run Segmentation and Post Process
The Segmentations results can be used to estimate the contact point of objects to ground surface.
```
python3 main.py --Dataset={src} --Segmenter={seg} --Segment --VisSegment
```
You can filter segmentation masks based on their confidence scores or class labels
```
python3 main.py --Dataset={src} --Segmenter={seg} --SegPostProc --VisSegment --SegTh=0.5 --classes_to_keep 2 5 7
```
| Flag | Function |
| ---- | -------- |
| `--classes_to_keep` | object classes to keep based on COCO format |
| `--SegTh` | filter based on confidence threshold of masks |

## Step 5: Object Detection and Post Process
```
python3 main.py --Dataset={src}  --Detector={det} --Detect --VisDetect
python3 main.py --Dataset={src}  --Detector={det} --VisDetect --DetPostProc --DetTh=0.5 --classes_to_keep 2 5 7
```
| Flag | Function |
| ---- | -------- |
| `--classes_to_keep` | object classes to keep based on COCO format |
| `--DetTh` | filter based on confidence threshold of bboxes |

you can also convert the detections to coco format for external use
```
python3 main.py --Dataset={src}  --Detector={det} --ConvertDetsToCOCO
```
You can get the contact point and backprojection points of the detections as below
```
python3 main.py --Dataset={src}  --Detector={det} --Segmenter={seg}  --Homography --BackprojectSource=detections\ --TopView=<TopView> --BackprojectionMethod=<Method> --ContactPoint=<CP>
```
| Flag | Function |
| ---- | -------- |
| `--Homography` | initiates backprojection module |
| `--BackprojectSource` | [detection, tracks] |
| `--TopView` | [GoogleMap, OrthoPhoto] |
| `--BackprojectionMethod` | [Homography, DSM] |
| `--ContactPoint` | [BottomPoint, Center, BottomSeg, LineSeg] |

To visualize the backprojected results
```
python3 main.py --Dataset={src}  --Detector={det} --VisContactPoint --VisCPTop --BackprojectSource=detections\
--TopView={tv} --BackprojectionMethod={bpm} --ContactPoint={cp}
```

## Step 6: Multi Object Tracking and Post Process
perform tracking by detection
```
python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --Track --VisTrack
```
back project tracks
```
python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --Homography --Meter --VisTrajectories --VisTrackTop
--BackprojectSource=tracks --TopView={tv}
```

Once the tracking is done you would be able to run a series of postprocessing steps on tracks.
```
python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --TrackPostProc --TrackTh={th} --RemoveInvalidTracks
--SelectDifEdgeInROI --SelectEndingInROI --SelectBeginInROI --MaskGPFrame --HasPointsInROI --MaskROI  --CrossROI --CrossROIMulti --JustEnterROI --JustExitROI --WithinROI --Interpolate --ExitOrCrossROI  --BackprojectSource=tracks --TopView=[GoogleMap/OrthoPhoto]
```
| Flag  | Function|
| ----- | ------- |
|`--TrackPostProc` | Flag to activate post processing module |
|`--TrackTh` | Resampling threshold on ground plane |
|`--Interpolate` | Interpolate tracks|
|`--RemoveInvalidTracks` | Remove tracks with only one ditection |
|`--SelectEndingInROI` | Select tracks that end in ROI |
|`--SelectBeginInROI` | Select tracks that start in ROI |
|`--MaskGPFrame` | Mask detections that are out of topview frame |
|`--HasPointsInROI` | Only keep tracks that has points in ROI |
|`--MaskROI`  | Mask detections that are out of ROI |
|`--CrossROI` | Keep tracks that cross ROI edge |
|`--CrossROIMulti` | Keep tracks that cross multiple ROI edges(>=2) |
|`--JustEnterROI` | Keep tracks that just enter ROI but do not exit |
|`--JustExitROI` | Keep tracks that exit ROi but do not enter |
|`--WithinROI` | Keep tracks that neither enter nor exit ROI |

## Step 7: Extract prototypes
There are two methods to extract prototype tracks
* Using "Tack Labelling GUI" in which a human annotator can select tracks
```
python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --TrackLabelingGUI --VisLabelledTrajectories
```
* Using automated track extraction based on ROI
```
python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --ExtractCommonTracks
--VisLabelledTrajectories --ResampleTH={th} --TopView={tp}
```

## Step 8: Trun Counts
At the end you would perform counting on the tracks using prototype tracks or prototype densities
```
python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --Count --CountMetric={metric}
--EvalCount --CacheCounter --CountVisDensity --ResampleTH={th} --TopView={tp}
```

if you are choosing kde as your metric, you would be able to optimize the kde bw prior to running counting with
```
python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --FindOptimalKDEBW --ResampleTH={th} --TopView={tp}
```

## Step 9: Visulize Results on Video
visualize all results with turn labels, id labels, ...
```
python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --CountMetric={met} --VisTrackMoI
```

## Step 10: Average Counts across sources
This is only applicable to multi camera envs
```
python3 main.py --MultiCam --Dataset={src}  --Detector={det} --Tracker={tra} --CountMetric={metric} --AverageCountsMC --EvalCountMC
```