function [ world_pt ] = image_to_world(homography_matrix,img_pt)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
load('GoProHero8CameraParams.mat')
img_pt = undistortFisheyePoints(img_pt,GoProHero8cameraParams.Intrinsics);
%img_pt = double([img_pt(1) img_pt(2)]);% ./  [704 480 1]';
img_pt = [ img_pt(1) img_pt(2) 1];% ./  [1920 1080 1]';
%world_pt = pointsToWorld(GoProcameraParamsF.Intrinsics,rmat',tvec,img_pt);
world_pt = img_pt*(homography_matrix)';
world_pt = world_pt ./ world_pt(end);
world_pt = [int32(world_pt(1)) int32(world_pt(2))];


end

