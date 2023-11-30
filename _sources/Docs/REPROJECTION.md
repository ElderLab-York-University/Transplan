# Backprojection Guide
This pipeline supports:
  * reprojection from homographies
  * reprojection with camera extrinsics and DSM terrain
  * reprojection on detecion or tracks level
  * topview from Google Map or OrthoPhoto

**NOTE**: before running any functionality of the pipeline makesure to format data according to [our requirements](./).

**NOTE**: there is a full-functionality template in `run.py`.

## Solve Homographies
This flag pops up a GUI by wich user can select pair points between topview and camera view.
The result of this GUI would be a planer transformation between the groundplane on cemra view and top view.
The `--VisHomographyGUI` will visualize this planer transformation.

```
python3 main.py --Dataset={src} --HomographyGUI --VisHomographyGUI --Frame=1 --TopView=GoogleMap/OrthoPhoto")
```
Results will be stored under src/Results/Visualization/*

See a sample video of working with GUI below

https://github.com/ElderLab-York-University/Transplan/blob/main/Docs/Assets/HomographyGUISample.mp4

## Backproject detections
The `--ContactPoint` will determine how to estimate object contact point to ground.
```
python3 main.py --Dataset={src}  --Detector={det} --Homography --BackprojectSource=detections --TopView=[GoogleMap/OrthoPhoto]\
 --BackprojectionMethod=[Homography/DSM] --ContactPoint=[BottomPoint/Center/BottomSeg/LineSeg]
```
Results will be stored under src/Results/Detection/*

## Visualize contact points
```
python3 main.py --Dataset={src}  --Detector={det} --VisContactPoint --BackprojectSource=detections --TopView=OrthoPhoto\
--BackprojectionMethod=Homography --ContactPoint=LineSeg
```
Results will be stored under src/Results/Visualization/*

## Visualize backprojected contact points
```
python3 main.py --Dataset={src}  --Detector={det} --VisCPTop --BackprojectSource=detections --TopView=OrthoPhoto\
--BackprojectionMethod=Homography --ContactPoint=LineSeg
```
Results will be stored under src/Results/Visualization/*

## Backproject tracks
```
python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --Homography --BackprojectSource=tracks\
--TopView=[GoogleMap/OrthoPhoto]
```
Results will be stored under src/Results/Tracking/*

## Visualize backprojected tracks
```
python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --VisTrajectories --VisTrackTop\
--BackprojectSource=tracks --TopView=[GoogleMap/OrthoPhoto]
```
Results will be stored under src/Results/Visualization/*
