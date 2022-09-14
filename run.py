import os
sources = ["./../Dataset/SOW_src2", "./../Dataset/SOW_src3", "./../Dataset/SOW_src4"]
# options :['./../Dataset/DandasStAtNinthLineFull', "./../Dataset/SOW_src1", "./../Dataset/SOW_src2"]
detectors = ["detectron2"]
trackers = ["gsort"] # ["sort", "CenterTrack", "DeepSort", "ByteTrack", "gsort", "OCSort"]
clusters = ["SpectralFull"]
metrics = ["cos", "tcos", "cmm", "ccmm", "tccmm", "hausdorff", "ptcos"] # options: ["cos", "tcos", "cmm", "ccmm", "tccmm", "hausdorff", "ptcos"]
for src in sources:
    # os.system(f"python3 main.py --Dataset={src}  --Detector=detectron2 --Tracker=sort --HomographyGUI --VisHomographyGUI")

    for det in detectors:
        print(f"detecting ----> src:{src} det:{det}")
        os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker=sort --Detect --DetPostProc --DetMask --DetTh=0.75 --VisDetect --VisROI")

    #     # os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker=sort --Detect --DetPostProc --DetMask --DetTh=0.75 --VisDetect --VisROI")

    # for det in detectors:
    #     for tra in trackers:
    #         print(f"tracking ---> src:{src} det:{det} tra:{tra}")
    #         os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --Track --VisTrack --Homography --Meter --TrackPostProc --TrackTh=8 --VisTrajectories")

            # os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --Track --VisTrack --Homography --Meter --TrackPostProc --TrackTh=8 --VisTrajectories")

    # for det in detectors:
    #     for tra in trackers:
    #         for met in metrics:
    #             for clt in clusters:
    #                 print(f"clustering ----> det:{det} tra:{tra} met:{met} clt:{clt}")
    #                 os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --Cluster --ClusteringAlgo={clt} --ClusterMetric={met}")
    
    # for det in detectors:
    #     for tra in trackers:
    #         os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --TrackLabelingGUI --VisLabelledTrajectories")

    # for det in detectors:
    #     for tra in trackers:
    #         for metric in metrics:
    #             print(f"counting metric:{metric}")
    #             os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --Count --CountMetric={metric} --EvalCount")

    # for det in detectors:
    #     for tra in trackers:
    #         for met in metrics[:1]:
    #             print(f"visualizing MOI -----> det:{det} tra:{tra} met:{met}")
    #             os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --CountMetric={met} --VisTrackMoI")
