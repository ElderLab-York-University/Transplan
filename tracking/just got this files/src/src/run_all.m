function [] = run_all(h)
for i = 55:66
    filename = strcat('faster_rcnn_inception_v2_coco_2018_01_28GXAB07',num2str(i),'.MP4_detection_test.csv');
    run_tracking(h, 'D:\projects\trans-plan\data\2020\May\351cm_height\', filename);
end
end