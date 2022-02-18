function [ img_pt ] = world_to_image(homography_matrix, world_pt, rmat, tvec )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

% img_pt = double([ img_pt(1) img_pt(2) 1]);% ./  [704 480 1]';
% % img_pt = double([ img_pt(1) img_pt(2) 1]') ./  [1920 1080 1]';
% 
% world_pt = img_pt*(homography_matrix)';
% world_pt = world_pt ./ world_pt(end);
% world_pt = [world_pt(1) world_pt(2)];

load('goproF.mat')
world_pt = double([world_pt(1) world_pt(2) 0]);                 
%img_pt = worldToImage(GoProcameraParamsF.Intrinsics,rmat',tvec,world_pt,'ApplyDistortion',true);
img_pt = world_pt*inv(homography_matrix');

%img_pt = img_pt .* [640 480 1]';
% img_pt = img_pt .* [1920 1080 1]';

img_pt = img_pt ./ img_pt(end);
img_pt = [int32(img_pt(1))  int32(img_pt(2))];

end

