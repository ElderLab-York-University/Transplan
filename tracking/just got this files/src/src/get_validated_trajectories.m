% Gong Cheng, Sep 2017.
% Output the trajectories with their maximum displacement inside the ROI

function [ ] = get_validated_trajectories( kalman_trajectories, validated_trajectories, distance_threshold, lifetime_threshold )
%GOPRO_KALMAN_TRAJECTORY_THRESHOLD_TEST_1 Summary of this function goes here
%   Detailed explanation goes here

%==========================================================================
% Pre-settings
%==========================================================================

% Load the tracks.
track_mat = load(kalman_trajectories);
track_file = track_mat.recorded_tracks;


%==========================================================================
% Show the points and the estimated trajectory
%==========================================================================

% Create an array to save the max_dis
dis_array = zeros(1,length(track_file));

recorded_tracks = [];

% Comparison
for ii = 1:length(track_file)
    % Get the tracking ID.
    tr_id = track_file(ii).id;
    
    % Get the discrete trajectory dots.
    tr_dots = track_file(ii).trajectory;
    
    % If the number of pixels inside the ROI is greater than 1, compute the
    % maximum displacement.
    if size(tr_dots,1) > lifetime_threshold   % threshold set by Gong is 1. I am experimenting with different values
        start_point = tr_dots(1,:);
        end_point = tr_dots(end,:);
        start_point = double(start_point);
        end_point = double(end_point);
        
%         dist_travelled = norm(end_point - start_point);
        dist_travelled = pdist2(end_point, start_point);
    else
        dist_travelled = 0;
    end

    dis_array(ii) = dist_travelled;
    




%     % If the number of pixels inside the ROI is greater than 1, compute the
%     % maximum displacement.
%     start_point = tr_dots(1,:);
%     end_point = tr_dots(end,:);
%     start_point = double(start_point);
%     end_point = double(end_point);
%     dist_travelled = norm(end_point - start_point);
% 
%     dis_array(ii) = dist_travelled;

%     % save the distances of tracks to a MAT file
%     save('distance_travelled.mat','dis_array');

        
    %-----------------------------------------------
    % Draw the maximum distance
    %-----------------------------------------------
    if dist_travelled > distance_threshold
        recorded_tracks = [recorded_tracks; track_file(ii)];
    end
    length(recorded_tracks)
    
end
    
    
    % Save the tracks to a MAT file.
save(validated_trajectories,'recorded_tracks');
fprintf('the overall number of tracks = %d  \n',length(recorded_tracks));
    
end

