function undistortedImage = generate_undistorted_image(calib_filename, original_image_filepath)
calib_params = load(calib_filename);
im = imread(original_image_filepath);
undistortedImage = undistortFisheyeImage(im, calib_params(1).Intrinsics);
end