import os
# choose the segment/dataset/video

segments = [
######### HW7 Train Segment
    # # # Seg03
    # "/mnt/data/HW7Leslie/Seg03",
    # # # Seg05
    # "/mnt/data/HW7Leslie/Seg05",
    # # # Seg06
    # "/mnt/data/HW7Leslie/Seg06",
    # #Seg07
    # "/mnt/data/HW7Leslie/Seg07",
    # #Seg08
    # "/mnt/data/HW7Leslie/Seg08",
    # # # Seg09
    # "/mnt/data/HW7Leslie/Seg09",
    # # # Seg11
    # "/mnt/data/HW7Leslie/Seg11",
    # # # Seg12
    # "/mnt/data/HW7Leslie/Seg12",
    # # # Seg13
    # "/mnt/data/HW7Leslie/Seg13",
    # #Seg15
    # "/mnt/data/HW7Leslie/Seg15",
    # #Seg20
    # "/mnt/data/HW7Leslie/Seg20",
    # #Seg21
    # "/mnt/data/HW7Leslie/Seg21",
    # #Seg22
    # "/mnt/data/HW7Leslie/Seg22",
    # #Seg24
    # "/mnt/data/HW7Leslie/Seg24",
    # #Seg27
    # "/mnt/data/HW7Leslie/Seg27",
    # #Seg29
    # "/mnt/data/HW7Leslie/Seg29",
    # #Seg30
    # "/mnt/data/HW7Leslie/Seg30",
    # #Seg31
    # "/mnt/data/HW7Leslie/Seg31",
######### HW7 Valid Segment
    # #Seg00
    # "/mnt/data/HW7Leslie/Seg00",
    # #Seg01
    # "/mnt/data/HW7Leslie/Seg01",
    # #Seg02
    # "/mnt/data/HW7Leslie/Seg02",
    # #Seg04
    # "/mnt/data/HW7Leslie/Seg04",
    # # # Seg10
    # "/mnt/data/HW7Leslie/Seg10",
    # #Seg14
    # "/mnt/data/HW7Leslie/Seg14",
    # #Seg16
    # "/mnt/data/HW7Leslie/Seg16",
    # #Seg17
    # "/mnt/data/HW7Leslie/Seg17",
    # #Seg18
    # "/mnt/data/HW7Leslie/Seg18",
    # #Seg19
    # "/mnt/data/HW7Leslie/Seg19",
    # #Seg23
    # "/mnt/data/HW7Leslie/Seg23",
    # #Seg25
    # "/mnt/data/HW7Leslie/Seg25",
    # #Seg26
    # "/mnt/data/HW7Leslie/Seg26",
    # #Seg28
    # "/mnt/data/HW7Leslie/Seg28",

]
sources = [
##### # TransPlan Dataset
    # # D9L
    # "/mnt/data/TransPlanData/Dataset/PreProcessedMain/D9L_Video1",
    # "/mnt/data/TransPlanData/Dataset/PreProcessedMain/D9L_Video2",
    # # DBR
    # "/mnt/data/TransPlanData/Dataset/PreProcessedMain/DBR_Video1",
    # "/mnt/data/TransPlanData/Dataset/PreProcessedMain/DBR_Video2",
    # # DWP
    # "/mnt/data/TransPlanData/Dataset/PreProcessedMain/DWP_Video1",
    # "/mnt/data/TransPlanData/Dataset/PreProcessedMain/DWP_Video2",
    # # ECR
    # "/mnt/data/TransPlanData/Dataset/PreProcessedMain/ECR_Video1",
    # "/mnt/data/TransPlanData/Dataset/PreProcessedMain/ECR_Video2",
##### # HW7 & Leslie  DATASET
##### # HW7 Train Split
    # # Seg03
    # "/mnt/data/HW7Leslie/Seg03/Seg03sc1",
    # "/mnt/data/HW7Leslie/Seg03/Seg03sc2",
    # "/mnt/data/HW7Leslie/Seg03/Seg03sc3",
    # "/mnt/data/HW7Leslie/Seg03/Seg03sc4",
    # # Seg05
    # "/mnt/data/HW7Leslie/Seg05/Seg05sc1",
    # "/mnt/data/HW7Leslie/Seg05/Seg05sc2",
    # "/mnt/data/HW7Leslie/Seg05/Seg05sc3",
    # "/mnt/data/HW7Leslie/Seg05/Seg05sc4",
    # # Seg06
    # "/mnt/data/HW7Leslie/Seg06/Seg06sc1",
    # "/mnt/data/HW7Leslie/Seg06/Seg06sc2",
    # "/mnt/data/HW7Leslie/Seg06/Seg06sc3",
    # "/mnt/data/HW7Leslie/Seg06/Seg06sc4",
    # # Seg07
    # "/mnt/data/HW7Leslie/Seg07/Seg07sc1",
    # "/mnt/data/HW7Leslie/Seg07/Seg07sc2",
    # "/mnt/data/HW7Leslie/Seg07/Seg07sc3",
    # "/mnt/data/HW7Leslie/Seg07/Seg07sc4",
    # # Seg08
    # "/mnt/data/HW7Leslie/Seg08/Seg08sc1",
    # "/mnt/data/HW7Leslie/Seg08/Seg08sc2",
    # "/mnt/data/HW7Leslie/Seg08/Seg08sc3",
    # "/mnt/data/HW7Leslie/Seg08/Seg08sc4",
    # # Seg09
    # "/mnt/data/HW7Leslie/Seg09/Seg09sc1",
    # "/mnt/data/HW7Leslie/Seg09/Seg09sc2",
    # "/mnt/data/HW7Leslie/Seg09/Seg09sc3",
    # "/mnt/data/HW7Leslie/Seg09/Seg09sc4",
    # # Seg11
    # "/mnt/data/HW7Leslie/Seg11/Seg11sc1",
    # "/mnt/data/HW7Leslie/Seg11/Seg11sc2",
    # "/mnt/data/HW7Leslie/Seg11/Seg11sc3",
    # "/mnt/data/HW7Leslie/Seg11/Seg11sc4",
    # # Seg12
    # "/mnt/data/HW7Leslie/Seg12/Seg12sc1",
    # "/mnt/data/HW7Leslie/Seg12/Seg12sc2",
    # "/mnt/data/HW7Leslie/Seg12/Seg12sc3",
    # "/mnt/data/HW7Leslie/Seg12/Seg12sc4",
    # # Seg13
    # "/mnt/data/HW7Leslie/Seg13/Seg13sc1",
    # "/mnt/data/HW7Leslie/Seg13/Seg13sc2",
    # "/mnt/data/HW7Leslie/Seg13/Seg13sc3",
    # "/mnt/data/HW7Leslie/Seg13/Seg13sc4",
    # # # Seg15
    # "/mnt/data/HW7Leslie/Seg15/Seg15sc1",
    # "/mnt/data/HW7Leslie/Seg15/Seg15sc2",
    # "/mnt/data/HW7Leslie/Seg15/Seg15sc3",
    # "/mnt/data/HW7Leslie/Seg15/Seg15sc4",
    # # Seg20
    # "/mnt/data/HW7Leslie/Seg20/Seg20sc1",
    # "/mnt/data/HW7Leslie/Seg20/Seg20sc2",
    # "/mnt/data/HW7Leslie/Seg20/Seg20sc3",
    # "/mnt/data/HW7Leslie/Seg20/Seg20sc4",
    # # Seg21
    # "/mnt/data/HW7Leslie/Seg21/Seg21sc1",
    # "/mnt/data/HW7Leslie/Seg21/Seg21sc2",
    # "/mnt/data/HW7Leslie/Seg21/Seg21sc3",
    # "/mnt/data/HW7Leslie/Seg21/Seg21sc4",
    # # Seg22
    # "/mnt/data/HW7Leslie/Seg22/Seg22sc1",
    # "/mnt/data/HW7Leslie/Seg22/Seg22sc2",
    # "/mnt/data/HW7Leslie/Seg22/Seg22sc3",
    # "/mnt/data/HW7Leslie/Seg22/Seg22sc4",
    # # Seg24
    # "/mnt/data/HW7Leslie/Seg24/Seg24sc1",
    # "/mnt/data/HW7Leslie/Seg24/Seg24sc2",
    # "/mnt/data/HW7Leslie/Seg24/Seg24sc3",
    # "/mnt/data/HW7Leslie/Seg24/Seg24sc4",
    # # Seg27
    # "/mnt/data/HW7Leslie/Seg27/Seg27sc1",
    # "/mnt/data/HW7Leslie/Seg27/Seg27sc2",
    # "/mnt/data/HW7Leslie/Seg27/Seg27sc3",
    # "/mnt/data/HW7Leslie/Seg27/Seg27sc4",
    # # Seg29
    # "/mnt/data/HW7Leslie/Seg29/Seg29sc1",
    # "/mnt/data/HW7Leslie/Seg29/Seg29sc2",
    # "/mnt/data/HW7Leslie/Seg29/Seg29sc3",
    # "/mnt/data/HW7Leslie/Seg29/Seg29sc4",
    # # Seg30
    # "/mnt/data/HW7Leslie/Seg30/Seg30sc1",
    # "/mnt/data/HW7Leslie/Seg30/Seg30sc2",
    # "/mnt/data/HW7Leslie/Seg30/Seg30sc3",
    # "/mnt/data/HW7Leslie/Seg30/Seg30sc4",
    # # Seg31
    # "/mnt/data/HW7Leslie/Seg31/Seg31sc1",
    # "/mnt/data/HW7Leslie/Seg31/Seg31sc2",
    # "/mnt/data/HW7Leslie/Seg31/Seg31sc3",
    # "/mnt/data/HW7Leslie/Seg31/Seg31sc4",
##### # HW7 & Leslie  DATASET
##### # HW7 Valid Split
    # # Seg00
    # "/mnt/data/HW7Leslie/Seg00/Seg00sc1",
    # "/mnt/data/HW7Leslie/Seg00/Seg00sc2",
    # "/mnt/data/HW7Leslie/Seg00/Seg00sc3",
    # "/mnt/data/HW7Leslie/Seg00/Seg00sc4",
    # # Seg01
    # "/mnt/data/HW7Leslie/Seg01/Seg01sc1",
    # "/mnt/data/HW7Leslie/Seg01/Seg01sc2",
    # "/mnt/data/HW7Leslie/Seg01/Seg01sc3",
    # "/mnt/data/HW7Leslie/Seg01/Seg01sc4",
    # # Seg02
    # "/mnt/data/HW7Leslie/Seg02/Seg02sc1",
    # "/mnt/data/HW7Leslie/Seg02/Seg02sc2",
    # "/mnt/data/HW7Leslie/Seg02/Seg02sc3",
    # "/mnt/data/HW7Leslie/Seg02/Seg02sc4",
    # # Seg04
    # "/mnt/data/HW7Leslie/Seg04/Seg04sc1",
    # "/mnt/data/HW7Leslie/Seg04/Seg04sc2",
    # "/mnt/data/HW7Leslie/Seg04/Seg04sc3",
    # "/mnt/data/HW7Leslie/Seg04/Seg04sc4",
    # # Seg10
    # "/mnt/data/HW7Leslie/Seg10/Seg10sc1",
    # "/mnt/data/HW7Leslie/Seg10/Seg10sc2",
    # "/mnt/data/HW7Leslie/Seg10/Seg10sc3",
    # "/mnt/data/HW7Leslie/Seg10/Seg10sc4",
    # # Seg14
    # "/mnt/data/HW7Leslie/Seg14/Seg14sc1",
    # "/mnt/data/HW7Leslie/Seg14/Seg14sc2",
    # "/mnt/data/HW7Leslie/Seg14/Seg14sc3",
    # "/mnt/data/HW7Leslie/Seg14/Seg14sc4",
    # # Seg16
    # "/mnt/data/HW7Leslie/Seg16/Seg16sc1",
    # "/mnt/data/HW7Leslie/Seg16/Seg16sc2",
    # "/mnt/data/HW7Leslie/Seg16/Seg16sc3",
    # "/mnt/data/HW7Leslie/Seg16/Seg16sc4",
    # # Seg17
    # "/mnt/data/HW7Leslie/Seg17/Seg17sc1",
    # "/mnt/data/HW7Leslie/Seg17/Seg17sc2",
    # "/mnt/data/HW7Leslie/Seg17/Seg17sc3",
    # "/mnt/data/HW7Leslie/Seg17/Seg17sc4",
    # # Seg18
    # "/mnt/data/HW7Leslie/Seg18/Seg18sc1",
    # "/mnt/data/HW7Leslie/Seg18/Seg18sc2",
    # "/mnt/data/HW7Leslie/Seg18/Seg18sc3",
    # "/mnt/data/HW7Leslie/Seg18/Seg18sc4",
    # # Seg19
    # "/mnt/data/HW7Leslie/Seg19/Seg19sc1",
    # "/mnt/data/HW7Leslie/Seg19/Seg19sc2",
    # "/mnt/data/HW7Leslie/Seg19/Seg19sc3",
    # "/mnt/data/HW7Leslie/Seg19/Seg19sc4",
    # # Seg23
    # "/mnt/data/HW7Leslie/Seg23/Seg23sc1",
    # "/mnt/data/HW7Leslie/Seg23/Seg23sc2",
    # "/mnt/data/HW7Leslie/Seg23/Seg23sc3",
    # "/mnt/data/HW7Leslie/Seg23/Seg23sc4",
    # # Seg25
    # "/mnt/data/HW7Leslie/Seg25/Seg25sc1",
    # "/mnt/data/HW7Leslie/Seg25/Seg25sc2",
    # "/mnt/data/HW7Leslie/Seg25/Seg25sc3",
    # "/mnt/data/HW7Leslie/Seg25/Seg25sc4",
    # # Seg26
    # "/mnt/data/HW7Leslie/Seg26/Seg26sc1",
    # "/mnt/data/HW7Leslie/Seg26/Seg26sc2",
    # "/mnt/data/HW7Leslie/Seg26/Seg26sc3",
    # "/mnt/data/HW7Leslie/Seg26/Seg26sc4",
    # # Seg28
    # "/mnt/data/HW7Leslie/Seg28/Seg28sc1",
    # "/mnt/data/HW7Leslie/Seg28/Seg28sc2",
    # "/mnt/data/HW7Leslie/Seg28/Seg28sc3",
    # "/mnt/data/HW7Leslie/Seg28/Seg28sc4",
]

cached_cnt_sources = [
# TransPlan Dataset
   # TransPlan Dataset
    # # D9L
    # "/mnt/data/TransPlanData/Dataset/PreProcessedMain/D9L_Video1",
    # "/mnt/data/TransPlanData/Dataset/PreProcessedMain/D9L_Video1",
    # # DBR
    # "/mnt/data/TransPlanData/Dataset/PreProcessedMain/DBR_Video1",
    # "/mnt/data/TransPlanData/Dataset/PreProcessedMain/DBR_Video1",
    # # DWP
    # "/mnt/data/TransPlanData/Dataset/PreProcessedMain/DWP_Video1",
    # "/mnt/data/TransPlanData/Dataset/PreProcessedMain/DWP_Video1",
    # # ECR
    # "/mnt/data/TransPlanData/Dataset/PreProcessedMain/ECR_Video1",
    # "/mnt/data/TransPlanData/Dataset/PreProcessedMain/ECR_Video1",
##### # HW7 & Leslie  DATASET
##### # HW7 Train Split
    # # Seg03
    # "/mnt/data/HW7Leslie/Seg03/Seg03sc1",
    # "/mnt/data/HW7Leslie/Seg03/Seg03sc2",
    # "/mnt/data/HW7Leslie/Seg03/Seg03sc3",
    # "/mnt/data/HW7Leslie/Seg03/Seg03sc4",
    # # Seg05
    # "/mnt/data/HW7Leslie/Seg05/Seg05sc1",
    # "/mnt/data/HW7Leslie/Seg05/Seg05sc2",
    # "/mnt/data/HW7Leslie/Seg05/Seg05sc3",
    # "/mnt/data/HW7Leslie/Seg05/Seg05sc4",
    # # Seg06
    # "/mnt/data/HW7Leslie/Seg06/Seg06sc1",
    # "/mnt/data/HW7Leslie/Seg06/Seg06sc2",
    # "/mnt/data/HW7Leslie/Seg06/Seg06sc3",
    # "/mnt/data/HW7Leslie/Seg06/Seg06sc4",
    # # Seg07
    # "/mnt/data/HW7Leslie/Seg07/Seg07sc1",
    # "/mnt/data/HW7Leslie/Seg07/Seg07sc2",
    # "/mnt/data/HW7Leslie/Seg07/Seg07sc3",
    # "/mnt/data/HW7Leslie/Seg07/Seg07sc4",
    # # Seg08
    # "/mnt/data/HW7Leslie/Seg08/Seg08sc1",
    # "/mnt/data/HW7Leslie/Seg08/Seg08sc2",
    # "/mnt/data/HW7Leslie/Seg08/Seg08sc3",
    # "/mnt/data/HW7Leslie/Seg08/Seg08sc4",
    # # Seg09
    # "/mnt/data/HW7Leslie/Seg09/Seg09sc1",
    # "/mnt/data/HW7Leslie/Seg09/Seg09sc2",
    # "/mnt/data/HW7Leslie/Seg09/Seg09sc3",
    # "/mnt/data/HW7Leslie/Seg09/Seg09sc4",
    # # Seg11
    # "/mnt/data/HW7Leslie/Seg11/Seg11sc1",
    # "/mnt/data/HW7Leslie/Seg11/Seg11sc2",
    # "/mnt/data/HW7Leslie/Seg11/Seg11sc3",
    # "/mnt/data/HW7Leslie/Seg11/Seg11sc4",
    # # Seg12
    # "/mnt/data/HW7Leslie/Seg12/Seg12sc1",
    # "/mnt/data/HW7Leslie/Seg12/Seg12sc2",
    # "/mnt/data/HW7Leslie/Seg12/Seg12sc3",
    # "/mnt/data/HW7Leslie/Seg12/Seg12sc4",
    # # Seg13
    # "/mnt/data/HW7Leslie/Seg13/Seg13sc1",
    # "/mnt/data/HW7Leslie/Seg13/Seg13sc2",
    # "/mnt/data/HW7Leslie/Seg13/Seg13sc3",
    # "/mnt/data/HW7Leslie/Seg13/Seg13sc4",
    # # # Seg15
    # "/mnt/data/HW7Leslie/Seg15/Seg15sc1",
    # "/mnt/data/HW7Leslie/Seg15/Seg15sc2",
    # "/mnt/data/HW7Leslie/Seg15/Seg15sc3",
    # "/mnt/data/HW7Leslie/Seg15/Seg15sc4",
    # # Seg20
    # "/mnt/data/HW7Leslie/Seg20/Seg20sc1",
    # "/mnt/data/HW7Leslie/Seg20/Seg20sc2",
    # "/mnt/data/HW7Leslie/Seg20/Seg20sc3",
    # "/mnt/data/HW7Leslie/Seg20/Seg20sc4",
    # # Seg21
    # "/mnt/data/HW7Leslie/Seg21/Seg21sc1",
    # "/mnt/data/HW7Leslie/Seg21/Seg21sc2",
    # "/mnt/data/HW7Leslie/Seg21/Seg21sc3",
    # "/mnt/data/HW7Leslie/Seg21/Seg21sc4",
    # # Seg22
    # "/mnt/data/HW7Leslie/Seg22/Seg22sc1",
    # "/mnt/data/HW7Leslie/Seg22/Seg22sc2",
    # "/mnt/data/HW7Leslie/Seg22/Seg22sc3",
    # "/mnt/data/HW7Leslie/Seg22/Seg22sc4",
    # # Seg24
    # "/mnt/data/HW7Leslie/Seg24/Seg24sc1",
    # "/mnt/data/HW7Leslie/Seg24/Seg24sc2",
    # "/mnt/data/HW7Leslie/Seg24/Seg24sc3",
    # "/mnt/data/HW7Leslie/Seg24/Seg24sc4",
    # # Seg27
    # "/mnt/data/HW7Leslie/Seg27/Seg27sc1",
    # "/mnt/data/HW7Leslie/Seg27/Seg27sc2",
    # "/mnt/data/HW7Leslie/Seg27/Seg27sc3",
    # "/mnt/data/HW7Leslie/Seg27/Seg27sc4",
    # # Seg29
    # "/mnt/data/HW7Leslie/Seg29/Seg29sc1",
    # "/mnt/data/HW7Leslie/Seg29/Seg29sc2",
    # "/mnt/data/HW7Leslie/Seg29/Seg29sc3",
    # "/mnt/data/HW7Leslie/Seg29/Seg29sc4",
    # # Seg30
    # "/mnt/data/HW7Leslie/Seg30/Seg30sc1",
    # "/mnt/data/HW7Leslie/Seg30/Seg30sc2",
    # "/mnt/data/HW7Leslie/Seg30/Seg30sc3",
    # "/mnt/data/HW7Leslie/Seg30/Seg30sc4",
    # # Seg31
    # "/mnt/data/HW7Leslie/Seg31/Seg31sc1",
    # "/mnt/data/HW7Leslie/Seg31/Seg31sc2",
    # "/mnt/data/HW7Leslie/Seg31/Seg31sc3",
    # "/mnt/data/HW7Leslie/Seg31/Seg31sc4",
##### # HW7 & Leslie  DATASET
##### # HW7 Valid Split
    # # Seg00
    # "/mnt/data/HW7Leslie/Seg00/Seg00sc1",
    # "/mnt/data/HW7Leslie/Seg00/Seg00sc2",
    # "/mnt/data/HW7Leslie/Seg00/Seg00sc3",
    # "/mnt/data/HW7Leslie/Seg00/Seg00sc4",
    # # Seg01
    # "/mnt/data/HW7Leslie/Seg01/Seg01sc1",
    # "/mnt/data/HW7Leslie/Seg01/Seg01sc2",
    # "/mnt/data/HW7Leslie/Seg01/Seg01sc3",
    # "/mnt/data/HW7Leslie/Seg01/Seg01sc4",
    # # Seg02
    # "/mnt/data/HW7Leslie/Seg02/Seg02sc1",
    # "/mnt/data/HW7Leslie/Seg02/Seg02sc2",
    # "/mnt/data/HW7Leslie/Seg02/Seg02sc3",
    # "/mnt/data/HW7Leslie/Seg02/Seg02sc4",
    # # Seg04
    # "/mnt/data/HW7Leslie/Seg04/Seg04sc1",
    # "/mnt/data/HW7Leslie/Seg04/Seg04sc2",
    # "/mnt/data/HW7Leslie/Seg04/Seg04sc3",
    # "/mnt/data/HW7Leslie/Seg04/Seg04sc4",
    # # Seg10
    # "/mnt/data/HW7Leslie/Seg10/Seg10sc1",
    # "/mnt/data/HW7Leslie/Seg10/Seg10sc2",
    # "/mnt/data/HW7Leslie/Seg10/Seg10sc3",
    # "/mnt/data/HW7Leslie/Seg10/Seg10sc4",
    # # Seg14
    # "/mnt/data/HW7Leslie/Seg14/Seg14sc1",
    # "/mnt/data/HW7Leslie/Seg14/Seg14sc2",
    # "/mnt/data/HW7Leslie/Seg14/Seg14sc3",
    # "/mnt/data/HW7Leslie/Seg14/Seg14sc4",
    # # Seg16
    # "/mnt/data/HW7Leslie/Seg16/Seg16sc1",
    # "/mnt/data/HW7Leslie/Seg16/Seg16sc2",
    # "/mnt/data/HW7Leslie/Seg16/Seg16sc3",
    # "/mnt/data/HW7Leslie/Seg16/Seg16sc4",
    # # Seg17
    # "/mnt/data/HW7Leslie/Seg17/Seg17sc1",
    # "/mnt/data/HW7Leslie/Seg17/Seg17sc2",
    # "/mnt/data/HW7Leslie/Seg17/Seg17sc3",
    # "/mnt/data/HW7Leslie/Seg17/Seg17sc4",
    # # Seg18
    # "/mnt/data/HW7Leslie/Seg18/Seg18sc1",
    # "/mnt/data/HW7Leslie/Seg18/Seg18sc2",
    # "/mnt/data/HW7Leslie/Seg18/Seg18sc3",
    # "/mnt/data/HW7Leslie/Seg18/Seg18sc4",
    # # Seg19
    # "/mnt/data/HW7Leslie/Seg19/Seg19sc1",
    # "/mnt/data/HW7Leslie/Seg19/Seg19sc2",
    # "/mnt/data/HW7Leslie/Seg19/Seg19sc3",
    # "/mnt/data/HW7Leslie/Seg19/Seg19sc4",
    # # Seg23
    # "/mnt/data/HW7Leslie/Seg23/Seg23sc1",
    # "/mnt/data/HW7Leslie/Seg23/Seg23sc2",
    # "/mnt/data/HW7Leslie/Seg23/Seg23sc3",
    # "/mnt/data/HW7Leslie/Seg23/Seg23sc4",
    # # Seg25
    # "/mnt/data/HW7Leslie/Seg25/Seg25sc1",
    # "/mnt/data/HW7Leslie/Seg25/Seg25sc2",
    # "/mnt/data/HW7Leslie/Seg25/Seg25sc3",
    # "/mnt/data/HW7Leslie/Seg25/Seg25sc4",
    # # Seg26
    # "/mnt/data/HW7Leslie/Seg26/Seg26sc1",
    # "/mnt/data/HW7Leslie/Seg26/Seg26sc2",
    # "/mnt/data/HW7Leslie/Seg26/Seg26sc3",
    # "/mnt/data/HW7Leslie/Seg26/Seg26sc4",
    # # Seg28
    # "/mnt/data/HW7Leslie/Seg28/Seg28sc1",
    # "/mnt/data/HW7Leslie/Seg28/Seg28sc2",
    # "/mnt/data/HW7Leslie/Seg28/Seg28sc3",
    # "/mnt/data/HW7Leslie/Seg28/Seg28sc4",
]

# choose the detectors
# options: ["GTHW7", "detectron2", "OpenMM", "YOLOv5", "YOLOv8", "InternImage"]
detectors = ["GTHW7"]

# choose the tracker
# options: ["GT", sort", "CenterTrack", "DeepSort", "ByteTrack", "gsort", "OCSort", "GByteTrack", "GDeepSort", "BOTSort", "StrongSort"]
trackers = ["ByteTrack"] 


# choose the clustering algorithm
# options: ["SpectralFull", "DBSCAN", "SpectralKNN"]
clusters = ["SpectralFull"]

# choose the metric for clustering and classification pqrt
# options: ["groi", "roi", "knn", "cos", "tcos", "cmm", "hausdorff", "kde",,"ccmm", "tccmm", "ptcos", "loskde", "hmmg"]
clt_metrics = ["tcos", "cmm"]
cnt_metrics = ["kde"]

# choose the segmenter
# options: ["InternImage"]
segmenters = ["InternImage"]

# for src, cached_cnt_pth in zip(sources, cached_cnt_sources):
    ########################################################
    # 0. extract images from video
    # os.system(f"python3 main.py --Dataset={src} --ExtractImages)
    ########################################################
    # print(f" extracting images from : {src}")
    # os.system(f"python3 main.py --Dataset={src} --ExtractImages")

    ########################################################
    # 1. estimate the Homography Metrix using Homography GUI 
    # os.system(f"python3 main.py --Dataset={src}  --Detector=detectron2 --Tracker=sort --HomographyGUI --VisHomographyGUI --Frame=1 --TopView=GoogleMap")
    ########################################################
    # print(f"src:{src}")
    # os.system(f"python3 main.py --Dataset={src}  --Detector=Null --Tracker=Null --VisHomographyGUI --TopView=OrthoPhoto")

    ########################################################
    # 2. visualizing the region of interest 
    # os.system(f"python3 main.py --Dataset={src}  --Detector=Null --Tracker=Null --VisROI --TopView=GoogleMap")
    #######################################################
    # print(f"src:{src}")
    # os.system(f"python3 main.py --Dataset={src}  --Detector=Null --Tracker=Null --VisROI --TopView=GoogleMap")

    #######################################################
    # 2.5 Segment Video Frames 
    #  os.system(f"python3 main.py --Dataset={src}  --Detector=Null --Tracker=Null --Segment --Segmenter={seg} --VisSegment --ForNFrames=2000")
    #######################################################
    # for seg in segmenters:
    #     print(f" Segmenting ----> src:{src} seg:{seg}")
    #     os.system(f"python3 main.py --Dataset={src}  --Detector=Null --Tracker=Null --Segmenter={seg} --Segment --VisSegment")

    #######################################################
    # 2.6 Segment Post Processing
    # os.system(f"python3 main.py --Dataset={src}  --Detector=Null --Tracker=Null --Segmenter={seg} --SegPostProc --VisSegment --SegTh=0.5 --classes_to_keep 2 5 7")
    #######################################################
    # for seg in segmenters:
    #     print(f" Segmenting ----> src:{src} seg:{seg}")
    #     os.system(f"python3 main.py --Dataset={src}  --Detector=Null --Tracker=Null --Segmenter={seg} --SegPostProc --VisSegment --SegTh=0.5 --classes_to_keep 2 5 7")

    ########################################################
    # 3. run the detection
    # the full commonad looks like : os.system(f"python3 main.py --Datas`et={src}  --Detector={det} --Tracker=NULL --Detect --VisDetect")
    ########################################################
    # for det in detectors:
    #     print(f"detecting ----> src:{src} det:{det}")
    #     os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker=NULL --Detect --VisDetect")

    ########################################################
    # 3.5 run the detection post processing
    # the full commonad looks like : os.system(f"python3 main.py --Datas`et={src}  --Detector={det} --Tracker=NULL --Detect --DetPostProc --DetMask --DetTh=0.50 --VisDetect")
    ########################################################
    # for det in detectors:
    #     print(f"detecting ----> src:{src} det:{det}")
    #     # os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker=NULL --DetPostProc --DetTh=0.5 --VisDetect")
    #     os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker=NULL --VisDetect --DetPostProc --DetTh=0.5 --classes_to_keep 2 5 7")

    ########################################################
    # 3.5.5 convert detections to coco format
    # the full commonad looks like : os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker=NULL")
    ########################################################
    # for det in detectors:
    #     print(f"converting to COCO ----> src:{src} det:{det}")
    #     os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker=NULL --ConvertDetsToCOCO")

    ########################################################
    # 3.6 back project detections
    # os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker=NULL --Homography --BackprojectSource=detections --TopView=[GoogleMap/OrthoPhoto] --BackprojectionMethod=[Homography/DSM] --ContactPoint=[BottomPoint/Center/BottomSeg/LineSeg]")
    ########################################################
    # for det in detectors:
    #     for seg in segmenters:
    #         print(f"detecting ----> src:{src} det:{det}")
    #         os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Segmenter={seg} --Tracker=NULL --Homography --BackprojectSource=detections --TopView=OrthoPhoto --BackprojectionMethod=Homography --ContactPoint=LineSeg")

    ########################################################
    # 3.7 Vis Contact Points and BP Points
    # os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker=NULL --VisContactPoint --BackprojectSource=detections --TopView=[GoogleMap/OrthoPhoto] --BackprojectionMethod=[Homography/DSM] --ContactPoint=[BottomPoint/Center/BottomSeg/LineSeg]")
    ########################################################
    # for det in detectors:
    #     print(f"detecting ----> src:{src} det:{det}")
    #     os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker=NULL --VisContactPoint --BackprojectSource=detections --TopView=OrthoPhoto --BackprojectionMethod=Homography --ContactPoint=LineSeg")
    #     os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker=NULL --VisCPTop        --BackprojectSource=detections --TopView=OrthoPhoto --BackprojectionMethod=Homography --ContactPoint=LineSeg")

    ########################################################
    # 4. run the tracking and backproject and convert to meter
    # os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --Track --VisTrack --ForNFrames=1800 --Homography --Meter --VisTrajectories --VisTrackTop")
    ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         print(f"tracking ---> src:{src} det:{det} tra:{tra}")
    #         # os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --Track --VisTrack --Homography --Meter --VisTrajectories --VisTrackTop --BackprojectSource=tracks --TopView=[GoogleMap/OrthoPhoto]")
    #         os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --Track --VisTrack --Homography --Meter --VisTrajectories --VisTrackTop --BackprojectSource=tracks --TopView=GoogleMap")

    ########################################################
    # 5. run the track post processing
    # os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --TrackPostProc --TrackTh=8 --RemoveInvalidTracks --SelectDifEdgeInROI --SelectEndingInROI --SelectBeginInROI --MaskGPFrame --HasPointsInROI --MaskROI  --CrossROI --CrossROIMulti --JustEnterROI --JustExitROI --WithinROI --Interpolate --ExitOrCrossROI  --BackprojectSource=tracks --TopView=[GoogleMap/OrthoPhoto]")
    ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         print(f"tracking POSTPROC ---> src:{src} det:{det} tra:{tra}")
    #         os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --TrackPostProc  --MaskGPFrame --HasPointsInROI --BackprojectSource=tracks --TopView=GoogleMap")
    #         os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --VisTrack --Homography --Meter --VisTrajectories --VisTrackTop --BackprojectSource=tracks --TopView=GoogleMap")

    ########################################################
    # 6. find optimum BW for kde fiting
    # os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --FindOptimalKDEBW --ResampleTH=2.0 --TopView=GoogleMap")
    ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         print(f"finding optimal bw for kde ---> src:{src} det:{det} tra:{tra}")
    #         os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --FindOptimalKDEBW --ResampleTH=2.0 --TopView=GoogleMap")

    ########################################################
    # 7. run clustering algorithm
    # apperently the clustering visulaizaiton is harcodded at the moment
    ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         for met in clt_metrics:
    #             for clt in clusters:
    #                 print(f"clustering ----> det:{det} tra:{tra} met:{met} clt:{clt}")
    #                 os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --ClusteringAlgo={clt} --ClusterMetric={met} --Cluster")
    
    ########################################################
    # 8. Run the track labelling GUI / go to 9.
    ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --TrackLabelingGUI --VisLabelledTrajectories")

    ########################################################
    # 9. Run automated track extraction and labelling
    ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         print(f"extract common tracks ----> det:{det}, tra:{tra}")
    #         os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --ExtractCommonTracks --VisLabelledTrajectories --ResampleTH=2.0 --TopView=GoogleMap")

    ########################################################
    # 10. Run the classification(counting) part
    # os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --Count --CountMetric={metric} --CountVisPrompt --EvalCount --UseCachedCounter --CachedCounterPth={cached_cnt_pth} --CacheCounter --CountVisDensity")
    ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         for metric in cnt_metrics:
    #             print(f"counting metric:{metric} det:{det} tra:{tra}")
    #             # os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --Count --CountMetric={metric} --EvalCount --CacheCounter --CountVisDensity --ResampleTH=2.0 --TopView=GoogleMap")
    #             # os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --Count --CountMetric={metric} --EvalCount --UseCachedCounter --CachedCounterPth={cached_cnt_pth} --TopView=GoogleMap")

    ########################################################
    # 11. Visualizing the results on a video including track label and track id
    # can be used to monitor the pipeline in detail
    ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         for met in cnt_metrics:
    #             print(f"visualizing MOI -----> det:{det} tra:{tra} met:{met}")
    #             os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --CountMetric={met} --VisTrackMoI")


    #_______________________MULTICAMERA_______________________#

# for src in segments:
#     print(f"running on seg:{src}")
    # ########################################################
    # # 0. Average Counts MC
    # # os.system(f"python3 main.py --MultiCam --Dataset={src}  --Detector={det} --Tracker={tra} -- --VisTrackTopMC")
    # ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         for metric in cnt_metrics:
    #             print(f"average MC counts ---> src:{src} det:{det} tra:{tra} cnt:{metric}")
    #             os.system(f"python3 main.py --MultiCam --Dataset={src}  --Detector={det} --Tracker={tra} --CountMetric={metric} --AverageCountsMC --EvalCountMC")