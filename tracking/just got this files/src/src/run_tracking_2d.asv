function [ ] = run_tracking_2d(detection_filename)
disp('Started matlab tracking')

%detection_filename = strcat(detection_folder,filename)
det_filename = split(detection_filename,'.');
output_trajectories = strcat(det_filename{1},'kalman_trajectories.mat')
validated_trajectories = strcat(det_filename{1},'validated_trajectories.mat')
start_points_file = strcat(det_filename{1},'start_points.mat')
end_points_file = strcat(det_filename{1},'end_points.mat')

%%% this parameter is 35 for tc1 and gopro. and 25 for the others
tracklet_non_assignment_cost = 25

%%% this parameter varies for each dataset
track_life = 1
dist_travelled = 50
% homography matrix
% homography_matrix = [3.87415, 0.19713,598.5754
%                     3.3230,     26.5289,-3462.7710
%                     0.00130, 0.010698, 1.        ];
%% tracking on the test data split
%detections = load(detection_filename);
detections = readtable(detection_filename);
if isempty(detections)
    quit
end

%%%homography_matrix = load(homography_matrix_fname)
detections.Properties.VariableNames={'fname','v','x','y','w','h','c'};
centroids = [detections.x+detections.w/2 detections.y+detections.h];
frame_num_centroids = [detections.fname centroids detections.x detections.y detections.w detections.h];
disp('running filter...')
kalman_filter_tracking(frame_num_centroids,output_trajectories,tracklet_non_assignment_cost);
disp('filter done...')
% %%% generate validated trajectories taking into account distance and lifetime threshold
get_validated_trajectories(output_trajectories,validated_trajectories, dist_travelled, track_life);
disp('generated trajectories')
% %%% get start and end points of the cars for clustering
generate_start_end_data( validated_trajectories, start_points_file, end_points_file)
disp('generated points')
% end