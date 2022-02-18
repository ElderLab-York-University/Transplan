% Hemanth Pidaparthy, Feb 2018.


function [ ] = kalman_filter_tracking(detections,output_trajectories,tracklet_non_assignment_cost)


% the previous versions accepted an ROI and tracked the objects while they 
% were within the ROI. Now we get rid of the ROI and track the object as 
% long as they are in the image.


%------------------------------------------------------
% Create an object to save all the video information
%------------------------------------------------------



%==========================================================================
% Parameter settings
%==========================================================================

% Cost of non-assignment 
con = tracklet_non_assignment_cost;

% Invisible for too long
iftl = 10;

% Age threshold
age_t = 1;

% Min visible count
mvc = 5;

% Initial estimate uncertainty variance [200, 50]
ini_est_error = [200, 50];

% Deviation of selected and actual model [100, 25]
motion_noise = [100, 25];

% Variance inaccuracy of detected location  1
measure_noise = 1;

%==========================================================================
% Tracking object setup
%==========================================================================
        
kalman_tracks = initializeTracks();
recorded_tracks = initializeTracks();
record_count = 1;

function tracks = initializeTracks()
    % create an empty array of tracks
    tracks = struct(...
        'id', {}, ...
        'bbox_list', {}, ...        
        'kalmanFilter', {}, ...
        'age', {}, ...
        'totalVisibleCount', {}, ...
        'consecutiveInvisibleCount', {}, ...
        'start_frame',{}, ...
        'end_frame',{}, ...
        'trajectory', {});
end

%==========================================================================
% Vehicle Tracking
%==========================================================================
        
nextId = 1; %ID of the next track]
%fnames = detections.fname;
% last_fname = fnames(end);
% last_number = split(last_fname,'.');
% last_number = last_number{1};
%last_number = length(detections);%str2num(last_number);
%all_centroids = [];
% Tracking the cars from the spcified start and end frames
% mapimg = imread("D:/projects/trans-plan/updated_algo/Refactored/transplan/drawnworld.png");
% v = VideoReader("D:/projects/trans-plan/data/2020/May/351cm_height/videos/GXAB0755.MP4");
for ii = 0:int32(detections(end,1))
    disp(ii)
    disp('of')
    disp(detections(end,1))
    centroids = [];
    bboxes = [];
    centroids = detections(detections(:,1)==ii,2:3);
    bboxes = int32(detections(detections(:,1)==ii,4:7));
    
    % Predict the location using Kalman filter.

    predictNewLocationsOfTracks();    


    % Data association using Hugarian algorithm.
    [assignments, unassignedTracks, unassignedDetections] = detectionToTrackAssignment();


    % Update the assigned tracks.

    updateAssignedTracks();


    % Update unassigned tracks.
    updateUnassignedTracks();
    
    %close all;
    %cost
    %centroids
%     frame = readFrame(v);
%     imshow(frame)
%     hold on
%     for t = 1:length(bboxes)
%         rectangle('Position',bboxes(t,:))
%         text(double(bboxes(t,1)),double(bboxes(t,2)),int2str(t))
%         mystr = strcat(num2str(centroids(t,1)),', ',num2str(centroids(t,2)));
%         text(double(bboxes(t,1)+5),double(bboxes(t,2)+5),mystr)
%     end
%     figure, imshow(mapimg)
%     hold on
%     for t = 1:length(kalman_tracks)
%         scatter(kalman_tracks(t).trajectory(:,1), kalman_tracks(t).trajectory(:,1))
%     end
    

    deleteLostTracks();
    createNewTracks(ii);
    
end

%-----------------------------------------------
% Save the MAT file
%-----------------------------------------------

% Save the tracks to a MAT file.
save(output_trajectories,'recorded_tracks');
%save('centroids.mat','all_centroids');



%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

function predictNewLocationsOfTracks()

    for jj = 1:length(kalman_tracks)
        
        %bbox = kalman_tracks(jj).bbox;

        % Predict the current location of the track.
        predictedCentroid = predict(kalman_tracks(jj).kalmanFilter);

        % Shift the bounding box so that its center is at
        % the predicted location.
        % subtract the appropriate height and width to get the 
        % top left points [x1 y1]. save bbox as [x1 y1 w h]
        %top_left_point = int32(predictedCentroid) - bbox(3:4) / 2;

        % since centroid is the bottom mid point, to get the top left
        % point, subtract the height and width accordingly
        %top_left_point = [int32(predictedCentroid(1))-bbox(3)/2  int32(predictedCentroid(2))-bbox(4)];
        
        %kalman_tracks(jj).bbox = [top_left_point, bbox(3:4)];
        %kalman_tracks(jj).bbox_list = [kalman_tracks(jj).bbox_list; kalman_tracks(jj).bbox];
    end
end

%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
function [assignments, unassignedTracks, unassignedDetections] = detectionToTrackAssignment()

    nTracks = length(kalman_tracks);
    nDetections = size(centroids, 1);

    % Compute the cost of assigning each detection to each track.
    cost = zeros(nTracks, nDetections);
    
    %------------------------------------------------------------------
    % This distance is too simple, we should use more
    % wise way to measure the difference.
    %------------------------------------------------------------------
    for i = 1:nTracks
        if length(centroids) ~=0
            diff = centroids - repmat(kalman_tracks(i).trajectory(end,:),[size(centroids,1),1]);
            cost(i, :) = sqrt(sum(diff .^ 2,2));
        end
    end
    
    
    % Solve the assignment problem.

    costOfNonAssignment = con;
    [assignments, unassignedTracks, unassignedDetections] = assignDetectionsToTracks(cost, costOfNonAssignment);
    
    
end

%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++        
        
function updateAssignedTracks()
    
    numAssignedTracks = size(assignments, 1);
    
    for i = 1:numAssignedTracks
            trackIdx = assignments(i, 1);
            detectionIdx = assignments(i, 2);
            centroid = centroids(detectionIdx, :);
            bbox = bboxes(detectionIdx, :);

            % Correct the estimate of the object's location
            % using the new detection.
            correct(kalman_tracks(trackIdx).kalmanFilter, centroid);

            % Replace predicted bounding box with detected
            % bounding box.
            kalman_tracks(trackIdx).bbox_list = [kalman_tracks(trackIdx).bbox_list;bbox];

            % Update track's age.
            kalman_tracks(trackIdx).age = kalman_tracks(trackIdx).age + 1;

            % Update visibility.
            kalman_tracks(trackIdx).totalVisibleCount = kalman_tracks(trackIdx).totalVisibleCount + 1;
            kalman_tracks(trackIdx).consecutiveInvisibleCount = 0;
            
            %----------------------------
            % Update the trajectory
            %----------------------------
            % have the trajectory point as the centroid
            trajectory_point = [centroid(1) centroid(2)];

            kalman_tracks(trackIdx).trajectory = [kalman_tracks(trackIdx).trajectory; trajectory_point];
            
    end
end

%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

function updateUnassignedTracks()
    for i = 1:length(unassignedTracks)
        ind = unassignedTracks(i);
        kalman_tracks(ind).age = kalman_tracks(ind).age + 1;
        kalman_tracks(ind).consecutiveInvisibleCount = ...
            kalman_tracks(ind).consecutiveInvisibleCount + 1;
    end
end

%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

function deleteLostTracks()
    if isempty(kalman_tracks)
        return;
    end

    invisibleForTooLong = iftl;
    ageThreshold = age_t;

    % Compute the fraction of the track's age for which it was visible.
    ages = [kalman_tracks(:).age];
    totalVisibleCounts = [kalman_tracks(:).totalVisibleCount];
    visibility = totalVisibleCounts ./ ages;

    % Find the indices of 'lost' tracks.
    lostInds = (ages < ageThreshold & visibility < 0.8) | ...
        [kalman_tracks(:).consecutiveInvisibleCount] >= invisibleForTooLong;
    
    
    % Get the lost tracks
    lost_track_ind = find(lostInds==1);
    
    % Save the lost tracks.
    for lost_ii = 1:length(lost_track_ind)
        recorded_tracks(record_count) = kalman_tracks(lost_track_ind(lost_ii));
        record_count = record_count+1;
    end    
    
    % Delete lost tracks.
    kalman_tracks = kalman_tracks(~lostInds); 
    
end

%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++        

function createNewTracks(frame_num)

    centroids = centroids(unassignedDetections, :);
    %bboxes = bboxes(unassignedDetections, :);

    for i = 1:size(centroids, 1)
            
        % Identify whehter the bounding box is inside of the ROI
        % If it's inside the ROI
        centroid = centroids(i,:);
        bbox = bboxes(i, :)
        
        trajectory = centroid;
        
        % Create a Kalman filter object.
        kalmanFilter = configureKalmanFilter('ConstantVelocity', ...
        centroid, ini_est_error, motion_noise, measure_noise);
        
        %if poly_intersection(bbox)==4
            % Create a new track.
            newTrack = struct(...
                'id', nextId, ...
                'bbox_list', bbox, ...
                'kalmanFilter', kalmanFilter, ...
                'age', 1, ...
                'totalVisibleCount', 1, ...
                'consecutiveInvisibleCount',0, ...
                'start_frame', frame_num,...
                'end_frame', 0,...
                'trajectory', trajectory);
        
            % Add it to the array of tracks.
            kalman_tracks(end + 1) = newTrack;

            % Increment the next id.
            nextId = nextId + 1;
       
        %end
    end
end

end
