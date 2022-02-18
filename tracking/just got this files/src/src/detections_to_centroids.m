function [XY_world_2D] = detections_to_centroids(detections, params, h)
XY = [detections.x+detections.w/2 detections.y+detections.h];
load(params);
XY_u = undistortFisheyePoints(XY, GoProHero8cameraParams.Intrinsics);
XY_u_ones = [XY_u ones(1, length(XY_u))'];
XY_world = XY_u_ones*h';
XY_world_norm = XY_world./XY_world(:,3);
XY_world_2D = XY_world_norm(:,1:2);
end