import os
# choose the dataset/video
# options : ['./../Dataset/DandasStAtNinthLineFull', './../Dataset/DandasStAtNinthLine', "./../Dataset/SOW_src1", "./../Dataset/SOW_src2", "./../Dataset/SOW_src3", "./../Dataset/SOW_src4"]
sources = [
# '/home/savoji/Desktop/TransPlanProject/Dataset/preprocessed/DundasStreetAtNinthLine/2020Dec18/GX010069'
# "./../Dataset/DandasStAtNinthLineFull"
'./../Dataset/DandasStAtNinthLine'
]

# choose the detectors
# options: ["detectron2", "OpenMM", "YOLOv5", "YOLOv8"]
detectors = ["DDETR"]

# choose the tracker
# options: ["sort", "CenterTrack", "DeepSort", "ByteTrack", "gsort", "OCSort", "GByteTrack", "GDeepSort"]
trackers = [ "ByteTrack"] 

# choose the clustering algorithm
# options: ["SpectralFull", "DBSCAN", "SpectralKNN"]
clusters = ["SpectralFull"]

# choose the metric for clustering and classification pqrt
# options: ["cos", "tcos", "cmm", "ccmm", "tccmm", "hausdorff", "ptcos", "loskde", "kde", "hmmg", "roi"]
metrics = ["cos", "tcos", "cmm", "ccmm", "tccmm", "hausdorff", "ptcos", "loskde", "kde","roi"]

for src in sources:
    ########################################################
    # 1. estimate the Homography Metrix using Homography GUI 
    ########################################################
    # os.system(f"python3 main.py --Dataset={src}  --Detector=detectron2 --Tracker=sort --HomographyGUI --VisHomographyGUI")

    ########################################################
    # 2. run the detection
    # the full commonad looks like : os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker=sort --Detect --DetPostProc --DetMask --DetTh=0.75 --VisDetect --VisROI")
    ########################################################
    for det in detectors:
        print(f"detecting ----> src:{src} det:{det}")
        os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker=sort  --Detect --VisDetect")

    ########################################################
    # 3. run the tracking 
    # os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --Track --VisTrack --Homography --Meter --TrackPostProc --TrackTh=8 --VisTrajectories")
    ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         print(f"tracking ---> src:{src} det:{det} tra:{tra}")
    #         os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --Homography --Meter --TrackPostProc --TrackTh=4 --TrackMask --VisTrajectories")

    ########################################################
    # 4. run clustering algorithm
    ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         for met in metrics:
    #             for clt in clusters:
    #                 print(f"clustering ----> det:{det} tra:{tra} met:{met} clt:{clt}")
    #                 os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --Cluster --ClusteringAlgo={clt} --ClusterMetric={met}")
    
    ########################################################
    # 5. Run the track labelling GUI / go to 6.
    ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --TrackLabelingGUI --VisLabelledTrajectories")

    ########################################################
    # 6. Run automated track extraction and labelling
    ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         print(f"extract common tracks ----> det:{det}, tra:{tra}")
    #         os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --ExtractCommonTracks --VisLabelledTrajectories")

    ########################################################
    # 7. Run the classification(counting) part
    # os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --Count --CountMetric={metric} --CountVisPrompt --EvalCount --UseCachedCounter --CacheCounter")
    ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         for metric in metrics:
    #             print(f"counting metric:{metric}")
    #             os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --Count --CountMetric={metric} --CacheCounter --EvalCount")

    ########################################################
    # 8. Visualizing the results on a video including track label and track id
    # can be used to monitor the pipeline in detail
    ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         for met in metrics[:1]:
    #             print(f"visualizing MOI -----> det:{det} tra:{tra} met:{met}")
    #             os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --CountMetric={met} --VisTrackMoI")
