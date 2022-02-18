
function [ ] = generate_start_end_data( validated_trajectories, start_points_file, end_points_file)


% Load the tracks.
track_mat = load(validated_trajectories);
track_file = track_mat.recorded_tracks;


start_points = [];
end_points = [];

% Get the start and end points for each track
for ii = 1:length(track_file)
    tra_id = track_file(ii).id;
   
    % Get the trajectory
    tra = double(track_file(ii).trajectory);
    p1 = [tra(1,1) tra(1,2)];
    p2 = [tra(end,1) tra(end,2)];
    
    
    %start_points = [start_points; world_to_image(p1, rmat, tvec)];
    %end_points = [end_points; world_to_image(p2, rmat, tvec)]; 
    start_points = [start_points; p1];
    end_points = [end_points; p2]; 
        
end


save(start_points_file,'start_points');
save(end_points_file,'end_points');
end
