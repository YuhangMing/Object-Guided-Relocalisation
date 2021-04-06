#include <ctime>
#include <chrono>
#include "input/azure.h"

namespace fusion
{

KinectAzure::KinectAzure(){
    // Get connected device count
    uint32_t device_count = k4a_device_get_installed_count();
    printf("Found %d connected devices:\n", device_count);

    // Open the first plugged in Kinect device
    if (K4A_FAILED(k4a_device_open(K4A_DEVICE_DEFAULT, &device)))
    {
        printf("Failed to open k4a device!\n");
        return;
    }

    // Test if the camera is working properly, by reading the serial number
    char *serial_number = NULL;
    size_t serial_number_length = 0;
    // get size of the serial number
    if (K4A_BUFFER_RESULT_TOO_SMALL != k4a_device_get_serialnum(device, NULL, &serial_number_length))
    {
        printf("%d: Failed to get serial number length\n", K4A_DEVICE_DEFAULT);
        k4a_device_close(device);
        device = NULL;
        return;
    }
    // allocate memory for the serial, then acquire it
    serial_number = (char*)(malloc(serial_number_length));
    if (serial_number == NULL)
    {
        printf("%d: Failed to allocate memory for serial number (%zu bytes)\n", K4A_DEVICE_DEFAULT, serial_number_length);
        k4a_device_close(device);
        device = NULL;
        return;
    }
    if (K4A_BUFFER_RESULT_SUCCEEDED != k4a_device_get_serialnum(device, serial_number, &serial_number_length))
    {
        printf("%d: Failed to get serial number\n", K4A_DEVICE_DEFAULT);
        free(serial_number);
        serial_number = NULL;
        k4a_device_close(device);
        device = NULL;
        return;
    }

    // Configure the input rgb and depth images
    k4a_device_configuration_t config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    // config.camera_fps = K4A_FRAMES_PER_SECOND_15;
    config.camera_fps = K4A_FRAMES_PER_SECOND_30;
    config.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
    // config.depth_mode = K4A_DEPTH_MODE_NFOV_2X2BINNED;
    // config.depth_mode = K4A_DEPTH_MODE_WFOV_UNBINNED;    // up to 15 fps
    // config.depth_mode = K4A_DEPTH_MODE_WFOV_2X2BINNED;
    config.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
    config.color_resolution = K4A_COLOR_RESOLUTION_720P;
    config.synchronized_images_only = true; // ensures that depth and color images are both available in the capture
    
    // Get the calibration
    if (K4A_RESULT_SUCCEEDED !=
        k4a_device_get_calibration(device, config.depth_mode, config.color_resolution, &calibration))
    {
        fprintf(stderr, "Failed to get calibration\n");
        k4a_device_close(device);
        return;
    }
    // print calibration informaion
    // {
    //     auto calib = calibration.color_camera_calibration;
    //     std::cout << "===== Device " << K4A_DEVICE_DEFAULT << ": " << serial_number << " =====\n";
    //     std::cout << "resolution width: " << calib.resolution_width << std::endl;
    //     std::cout << "resolution height: " << calib.resolution_height << std::endl;
    //     std::cout << "principal point x: " << calib.intrinsics.parameters.param.cx << std::endl;
    //     std::cout << "principal point y: " << calib.intrinsics.parameters.param.cy << std::endl;
    //     std::cout << "focal length x: " << calib.intrinsics.parameters.param.fx << std::endl;
    //     std::cout << "focal length y: " << calib.intrinsics.parameters.param.fy << std::endl;
    //     std::cout << "radial distortion coefficients:" << std::endl;
    //     std::cout << "k1: " << calib.intrinsics.parameters.param.k1 << std::endl;
    //     std::cout << "k2: " << calib.intrinsics.parameters.param.k2 << std::endl;
    //     std::cout << "k3: " << calib.intrinsics.parameters.param.k3 << std::endl;
    //     std::cout << "k4: " << calib.intrinsics.parameters.param.k4 << std::endl;
    //     std::cout << "k5: " << calib.intrinsics.parameters.param.k5 << std::endl;
    //     std::cout << "k6: " << calib.intrinsics.parameters.param.k6 << std::endl;
    //     std::cout << "center of distortion in Z=1 plane, x: " << calib.intrinsics.parameters.param.codx << std::endl;
    //     std::cout << "center of distortion in Z=1 plane, y: " << calib.intrinsics.parameters.param.cody << std::endl;
    //     std::cout << "tangential distortion coefficient x: " << calib.intrinsics.parameters.param.p1 << std::endl;
    //     std::cout << "tangential distortion coefficient y: " << calib.intrinsics.parameters.param.p2 << std::endl;
    //     std::cout << "metric radius: " << calib.intrinsics.parameters.param.metric_radius << std::endl;
    //     std::cout << "==================================\n";
    //     calib = calibration.depth_camera_calibration;
    //     std::cout << "resolution width: " << calib.resolution_width << std::endl;
    //     std::cout << "resolution height: " << calib.resolution_height << std::endl;
    //     std::cout << "principal point x: " << calib.intrinsics.parameters.param.cx << std::endl;
    //     std::cout << "principal point y: " << calib.intrinsics.parameters.param.cy << std::endl;
    //     std::cout << "focal length x: " << calib.intrinsics.parameters.param.fx << std::endl;
    //     std::cout << "focal length y: " << calib.intrinsics.parameters.param.fy << std::endl;
    //     std::cout << "radial distortion coefficients:" << std::endl;
    //     std::cout << "k1: " << calib.intrinsics.parameters.param.k1 << std::endl;
    //     std::cout << "k2: " << calib.intrinsics.parameters.param.k2 << std::endl;
    //     std::cout << "k3: " << calib.intrinsics.parameters.param.k3 << std::endl;
    //     std::cout << "k4: " << calib.intrinsics.parameters.param.k4 << std::endl;
    //     std::cout << "k5: " << calib.intrinsics.parameters.param.k5 << std::endl;
    //     std::cout << "k6: " << calib.intrinsics.parameters.param.k6 << std::endl;
    //     std::cout << "center of distortion in Z=1 plane, x: " << calib.intrinsics.parameters.param.codx << std::endl;
    //     std::cout << "center of distortion in Z=1 plane, y: " << calib.intrinsics.parameters.param.cody << std::endl;
    //     std::cout << "tangential distortion coefficient x: " << calib.intrinsics.parameters.param.p1 << std::endl;
    //     std::cout << "tangential distortion coefficient y: " << calib.intrinsics.parameters.param.p2 << std::endl;
    //     std::cout << "metric radius: " << calib.intrinsics.parameters.param.metric_radius << std::endl;
    //     std::cout << "==================================\n";
    // }

    // Get the transformation
    transformation = k4a_transformation_create(&calibration);

    // Start the camera with the given configuration
    if (K4A_RESULT_SUCCEEDED != k4a_device_start_cameras(device, &config))
    {
        printf("Failed to start device\n");
        k4a_device_close(device);
        return;
    }

    // Prepare for the pinhole camera model
    pinhole = create_pinhole_from_xy_range(&calibration, K4A_CALIBRATION_TYPE_COLOR);
    k4a_image_create(K4A_IMAGE_FORMAT_CUSTOM, pinhole.width, pinhole.height,
                        pinhole.width * (int)sizeof(coordinate_t), &lut);
    create_undistortion_lut(&calibration, K4A_CALIBRATION_TYPE_COLOR,
                            &pinhole, lut, INTERPOLATION_NEARESTNEIGHBOR);
    k4a_image_create(K4A_IMAGE_FORMAT_COLOR_BGRA32, pinhole.width, pinhole.height,
                        pinhole.width * (int)sizeof(uint32_t), &undistorted_color_image);
    k4a_image_create(K4A_IMAGE_FORMAT_DEPTH16, pinhole.width, pinhole.height,
                        pinhole.width * (int)sizeof(uint16_t), &undistorted_depth_image);
}

KinectAzure::~KinectAzure(){
    // Release the images and the capture
    k4a_image_release(color_image);
    k4a_image_release(depth_image);
    // k4a_image_release(ir_image);
    // k4a_image_release(transformed_color_image);
    k4a_image_release(transformed_depth_image);
    // k4a_image_release(transformed_depth_image_downscaled);
    k4a_image_release(lut);
    k4a_image_release(undistorted_color_image);
    k4a_image_release(undistorted_depth_image);
    k4a_capture_release(capture);
    // Shut down the camera when finished with application logic
    k4a_device_stop_cameras(device);
    k4a_device_close(device);
}

bool KinectAzure::getNextPair(cv::Mat &mRGB, cv::Mat &mDepth){
    // make a capture
    const int32_t TIMEOUT_IN_MS = 1000;
    switch (k4a_device_get_capture(device, &capture, TIMEOUT_IN_MS))
    {
    case K4A_WAIT_RESULT_SUCCEEDED:
        break;
    case K4A_WAIT_RESULT_TIMEOUT:
        printf("Timed out waiting for a capture\n");
        return false;
    case K4A_WAIT_RESULT_FAILED:
        printf("Failed to read a capture\n");
        return false;
    }

    // retrieve color image
    color_image = k4a_capture_get_color_image(capture);
    if (color_image == NULL)
    {
        printf("Failed to get color image from capture\n");
        return false;  
    } 
    // print_k4a_image_info(color_image, "[rgb]");
    // retrieve depth image
    depth_image = k4a_capture_get_depth_image(capture);
    if (depth_image == NULL)
    {
        printf("Failed to get depth image from capture\n");
        return false;
    }
    // print_k4a_image_info(depth_image, "[depth]");
    // // retrieve ir image
    // ir_image = k4a_capture_get_depth_image(capture);
    // if (ir_image == 0)
    // {
    //     printf("Failed to get ir image from capture\n");
    //     return false;
    // }
    // int ir_height = k4a_image_get_height_pixels(ir_image),
    //     ir_width = k4a_image_get_width_pixels(ir_image);
    // std::cout << "[ir] " << "\n"
    //           << "- format: " << k4a_image_get_format(ir_image) << "\n"
    //           << "- device_timestamp: " << k4a_image_get_device_timestamp_usec(ir_image) << "\n"
    //           << "- system_timestamp: " << k4a_image_get_system_timestamp_nsec(ir_image) << "\n"
    //           << "- resolution: " << ir_height << "x" << ir_width 
    //           << ", with stride: " << k4a_image_get_stride_bytes(ir_image)
    //           << std::endl;

    // // register color image to the depth image resolution
    // if (K4A_RESULT_SUCCEEDED != k4a_image_create(K4A_IMAGE_FORMAT_COLOR_BGRA32,
    //                                              k4a_image_get_width_pixels(depth_image), k4a_image_get_height_pixels(depth_image),
    //                                              k4a_image_get_width_pixels(depth_image) * 4 * (int)sizeof(uint8_t),
    //                                              &transformed_color_image))
    // {
    //     printf("Failed to create transformed depth image\n");
    //     return false;
    // }
    // if(K4A_RESULT_SUCCEEDED != 
    //    k4a_transformation_color_image_to_depth_camera(transformation, depth_image, color_image, transformed_color_image))
    // {
    //     fprintf(stderr, "Failed to transform the color image\n");
    //     return false;
    // }
    // print_k4a_image_info(transformed_color_image, "[transformed color]");
    
    // register depth image to the color image
    if (K4A_RESULT_SUCCEEDED != k4a_image_create(K4A_IMAGE_FORMAT_DEPTH16,
                                                 k4a_image_get_width_pixels(color_image), 
                                                 k4a_image_get_height_pixels(color_image),
                                                 k4a_image_get_width_pixels(color_image) * (int)sizeof(uint16_t),
                                                 &transformed_depth_image))
    {
        printf("Failed to create transformed depth image\n");
        return false;
    }
    if(K4A_RESULT_SUCCEEDED != 
       k4a_transformation_depth_image_to_color_camera(transformation, depth_image, transformed_depth_image))
    {
        fprintf(stderr, "Failed to transform the depth image\n");
        return false;
    }
    // print_k4a_image_info(transformed_depth_image, "[transformed depth]");

    // // downscale the color image -> CHECKED OUT, no problem
    // k4a_calibration_t calibration_color_downscaled;
    // memcpy(&calibration_color_downscaled, &calibration, sizeof(k4a_calibration_t));
    // calibration_color_downscaled.color_camera_calibration.resolution_width /= 2;
    // calibration_color_downscaled.color_camera_calibration.resolution_height /= 2;
    // calibration_color_downscaled.color_camera_calibration.intrinsics.parameters.param.cx /= 2;
    // calibration_color_downscaled.color_camera_calibration.intrinsics.parameters.param.cy /= 2;
    // calibration_color_downscaled.color_camera_calibration.intrinsics.parameters.param.fx /= 2;
    // calibration_color_downscaled.color_camera_calibration.intrinsics.parameters.param.fy /= 2;
    // k4a_transformation_t transformation_color_downscaled = k4a_transformation_create(&calibration_color_downscaled);
    // k4a_image_t color_image_downscaled = downscale_image_2x2_binning(color_image);
    // // register depth image to downscaled color image
    // k4a_image_t transformed_depth_image_downscaled;
    // if (K4A_RESULT_SUCCEEDED != k4a_image_create(K4A_IMAGE_FORMAT_DEPTH16,
    //                                              k4a_image_get_width_pixels(color_image_downscaled), 
    //                                              k4a_image_get_height_pixels(color_image_downscaled),
    //                                              k4a_image_get_width_pixels(color_image_downscaled) * (int)sizeof(uint16_t),
    //                                              &transformed_depth_image_downscaled))
    // {
    //     printf("Failed to create transformed depth image\n");
    //     return false;
    // }
    // if(K4A_RESULT_SUCCEEDED != 
    //    k4a_transformation_depth_image_to_color_camera(transformation_color_downscaled, depth_image, transformed_depth_image_downscaled))
    // {
    //     fprintf(stderr, "Failed to transform the depth image\n");
    //     return false;
    // }
    // print_k4a_image_info(transformed_depth_image_downscaled, "[downscaled transformed depth]");

    // undistort the color and depth image -> CHECKED OUT, no problem
    remap<uint32_t>(color_image, lut, undistorted_color_image, INTERPOLATION_NEARESTNEIGHBOR);
    // print_k4a_image_info(undistorted_color_image, "[undistorted color image]");
    remap<uint16_t>(transformed_depth_image, lut, undistorted_depth_image, INTERPOLATION_NEARESTNEIGHBOR);
    // print_k4a_image_info(undistorted_depth_image, "[undistorted depth image]");

    // Change from k4a_image_t to cv::Mat
    k4a_image_to_cv_mat(undistorted_color_image, mRGB, 1);
    k4a_image_to_cv_mat(undistorted_depth_image, mDepth, 0);
    // crop to 640*480
    cv::Rect myROI(320, 120, 640, 480);
    mRGB = mRGB(myROI);
    mDepth = mDepth(myROI);

    // // normalize for the purpose of visualization
    // cv::Mat bgr, scaledDepth;
    // cv::cvtColor(mRGB, bgr, CV_RGB2BGR);
    // normalize(mDepth, scaledDepth, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    // // scaledDepth.convertTo(scaledDepth, CV_8U, 1);
    // cv::imshow("RGB", bgr);
    // cv::imshow("Depth", scaledDepth);
    // int k = cv::waitKey(0);

    // time = std::chrono::duration_cast<std::chrono::milliseconds>(
    //     std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
    // id = frameIdx++;
    
    return true;
}

void KinectAzure::k4a_image_to_cv_mat(const k4a_image_t kImg, cv::Mat &mImg, int typeFlag)
{
    int width = k4a_image_get_width_pixels(kImg);
    int height = k4a_image_get_height_pixels(kImg);

    if(typeFlag == 0)
    {
        uint16_t *depth_buffer = reinterpret_cast<uint16_t *>(k4a_image_get_buffer(kImg));
        mImg = cv::Mat(height, width, CV_16UC1, depth_buffer);
    }
    else if(typeFlag == 1)
    {
        uint32_t *color_buffer = reinterpret_cast<uint32_t *>(k4a_image_get_buffer(kImg));
        mImg = cv::Mat(height, width, CV_8UC4, (void *)color_buffer);
        cv::cvtColor(mImg, mImg, CV_BGRA2RGB);
    } 
    else
    {
        std::cout << "!!!! Unsupported type flag, use 0 for depth image and 1 for color image." << std::endl;
        return;
    }
    
}

void KinectAzure::print_k4a_image_info(const k4a_image_t kImg, std::string str)
{
    std::cout << str << "\n"
              << "- format: " << k4a_image_get_format(kImg) << "\n"
              << "- device_timestamp: " << k4a_image_get_device_timestamp_usec(kImg) << "\n"
              << "- system_timestamp: " << k4a_image_get_system_timestamp_nsec(kImg) << "\n"
              << "- resolution: " << k4a_image_get_height_pixels(kImg) 
              << "x" << k4a_image_get_width_pixels(kImg) 
              << ", with stride: " << k4a_image_get_stride_bytes(kImg)
              << std::endl;
}

// For functions below:
// Copyright (c) Microsoft Corporation. 
// All rights reserved. Licensed under the MIT License.
k4a_image_t KinectAzure::downscale_image_2x2_binning(const k4a_image_t color_image)
{
    int color_image_width_pixels = k4a_image_get_width_pixels(color_image);
    int color_image_height_pixels = k4a_image_get_height_pixels(color_image);
    int color_image_downscaled_width_pixels = color_image_width_pixels / 2;
    int color_image_downscaled_height_pixels = color_image_height_pixels / 2;
    k4a_image_t color_image_downscaled = NULL;
    if (K4A_RESULT_SUCCEEDED != k4a_image_create(K4A_IMAGE_FORMAT_COLOR_BGRA32,
                                                 color_image_downscaled_width_pixels,
                                                 color_image_downscaled_height_pixels,
                                                 color_image_downscaled_width_pixels * 4 * (int)sizeof(uint8_t),
                                                 &color_image_downscaled))
    {
        printf("Failed to create downscaled color image\n");
        return color_image_downscaled;
    }

    uint8_t *color_image_data = k4a_image_get_buffer(color_image);
    uint8_t *color_image_downscaled_data = k4a_image_get_buffer(color_image_downscaled);
    for (int j = 0; j < color_image_downscaled_height_pixels; j++)
    {
        for (int i = 0; i < color_image_downscaled_width_pixels; i++)
        {
            int index_downscaled = j * color_image_downscaled_width_pixels + i;
            int index_tl = (j * 2 + 0) * color_image_width_pixels + i * 2 + 0;
            int index_tr = (j * 2 + 0) * color_image_width_pixels + i * 2 + 1;
            int index_bl = (j * 2 + 1) * color_image_width_pixels + i * 2 + 0;
            int index_br = (j * 2 + 1) * color_image_width_pixels + i * 2 + 1;

            color_image_downscaled_data[4 * index_downscaled + 0] = (uint8_t)(
                (color_image_data[4 * index_tl + 0] + color_image_data[4 * index_tr + 0] +
                 color_image_data[4 * index_bl + 0] + color_image_data[4 * index_br + 0]) /
                4.0f);
            color_image_downscaled_data[4 * index_downscaled + 1] = (uint8_t)(
                (color_image_data[4 * index_tl + 1] + color_image_data[4 * index_tr + 1] +
                 color_image_data[4 * index_bl + 1] + color_image_data[4 * index_br + 1]) /
                4.0f);
            color_image_downscaled_data[4 * index_downscaled + 2] = (uint8_t)(
                (color_image_data[4 * index_tl + 2] + color_image_data[4 * index_tr + 2] +
                 color_image_data[4 * index_bl + 2] + color_image_data[4 * index_br + 2]) /
                4.0f);
            color_image_downscaled_data[4 * index_downscaled + 3] = (uint8_t)(
                (color_image_data[4 * index_tl + 3] + color_image_data[4 * index_tr + 3] +
                 color_image_data[4 * index_bl + 3] + color_image_data[4 * index_br + 3]) /
                4.0f);
        }
    }

    return color_image_downscaled;
}

// Compute a conservative bounding box on the unit plane in which all the points have valid projections
void KinectAzure::compute_xy_range(const k4a_calibration_t *calibration, const k4a_calibration_type_t camera,
                                   const int width, const int height, float &x_min, float &x_max,
                                   float &y_min, float &y_max)
{
    // Step outward from the centre point until we find the bounds of valid projection
    const float step_u = 0.25f;
    const float step_v = 0.25f;
    const float min_u = 0;
    const float min_v = 0;
    const float max_u = (float)width - 1;
    const float max_v = (float)height - 1;
    const float center_u = 0.5f * width;
    const float center_v = 0.5f * height;

    int valid;
    k4a_float2_t p;
    k4a_float3_t ray;

    // search x_min
    for (float uv[2] = { center_u, center_v }; uv[0] >= min_u; uv[0] -= step_u)
    {
        p.xy.x = uv[0];
        p.xy.y = uv[1];
        k4a_calibration_2d_to_3d(calibration, &p, 1.f, camera, camera, &ray, &valid);

        if (!valid)
        {
            break;
        }
        x_min = ray.xyz.x;
    }

    // search x_max
    for (float uv[2] = { center_u, center_v }; uv[0] <= max_u; uv[0] += step_u)
    {
        p.xy.x = uv[0];
        p.xy.y = uv[1];
        k4a_calibration_2d_to_3d(calibration, &p, 1.f, camera, camera, &ray, &valid);

        if (!valid)
        {
            break;
        }
        x_max = ray.xyz.x;
    }

    // search y_min
    for (float uv[2] = { center_u, center_v }; uv[1] >= min_v; uv[1] -= step_v)
    {
        p.xy.x = uv[0];
        p.xy.y = uv[1];
        k4a_calibration_2d_to_3d(calibration, &p, 1.f, camera, camera, &ray, &valid);

        if (!valid)
        {
            break;
        }
        y_min = ray.xyz.y;
    }

    // search y_max
    for (float uv[2] = { center_u, center_v }; uv[1] <= max_v; uv[1] += step_v)
    {
        p.xy.x = uv[0];
        p.xy.y = uv[1];
        k4a_calibration_2d_to_3d(calibration, &p, 1.f, camera, camera, &ray, &valid);

        if (!valid)
        {
            break;
        }
        y_max = ray.xyz.y;
    }
}

pinhole_t KinectAzure::create_pinhole_from_xy_range(const k4a_calibration_t *calibration, 
                                                    const k4a_calibration_type_t camera)
{
    int width = calibration->depth_camera_calibration.resolution_width;
    int height = calibration->depth_camera_calibration.resolution_height;
    if (camera == K4A_CALIBRATION_TYPE_COLOR)
    {
        width = calibration->color_camera_calibration.resolution_width;
        height = calibration->color_camera_calibration.resolution_height;
    }

    float x_min = 0, x_max = 0, y_min = 0, y_max = 0;
    compute_xy_range(calibration, camera, width, height, x_min, x_max, y_min, y_max);

    pinhole_t pinhole;

    float fx = 1.f / (x_max - x_min);
    float fy = 1.f / (y_max - y_min);
    float px = -x_min * fx;
    float py = -y_min * fy;

    pinhole.fx = fx * width;
    pinhole.fy = fy * height;
    pinhole.px = px * width;
    pinhole.py = py * height;
    pinhole.width = width;
    pinhole.height = height;

    return pinhole;
}

void KinectAzure::create_undistortion_lut(const k4a_calibration_t *calibration, const k4a_calibration_type_t camera,
                                          const pinhole_t *pinhole, k4a_image_t lut, interpolation_t type)
{
    coordinate_t *lut_data = (coordinate_t *)(void *)k4a_image_get_buffer(lut);

    k4a_float3_t ray;
    ray.xyz.z = 1.f;

    int src_width = calibration->depth_camera_calibration.resolution_width;
    int src_height = calibration->depth_camera_calibration.resolution_height;
    if (camera == K4A_CALIBRATION_TYPE_COLOR)
    {
        src_width = calibration->color_camera_calibration.resolution_width;
        src_height = calibration->color_camera_calibration.resolution_height;
    }

    for (int y = 0, idx = 0; y < pinhole->height; y++)
    {
        ray.xyz.y = ((float)y - pinhole->py) / pinhole->fy;

        for (int x = 0; x < pinhole->width; x++, idx++)
        {
            ray.xyz.x = ((float)x - pinhole->px) / pinhole->fx;

            k4a_float2_t distorted;
            int valid;
            k4a_calibration_3d_to_2d(calibration, &ray, camera, camera, &distorted, &valid);

            coordinate_t src;
            if (type == INTERPOLATION_NEARESTNEIGHBOR)
            {
                // Remapping via nearest neighbor interpolation
                src.x = (int)floorf(distorted.xy.x + 0.5f);
                src.y = (int)floorf(distorted.xy.y + 0.5f);
            }
            else if (type == INTERPOLATION_BILINEAR || type == INTERPOLATION_BILINEAR_DEPTH)
            {
                // Remapping via bilinear interpolation
                src.x = (int)floorf(distorted.xy.x);
                src.y = (int)floorf(distorted.xy.y);
            }
            else
            {
                printf("Unexpected interpolation type!\n");
                exit(-1);
            }

            if (valid && src.x >= 0 && src.x < src_width && src.y >= 0 && src.y < src_height)
            {
                lut_data[idx] = src;

                if (type == INTERPOLATION_BILINEAR || type == INTERPOLATION_BILINEAR_DEPTH)
                {
                    // Compute the floating point weights, using the distance from projected point src to the
                    // image coordinate of the upper left neighbor
                    float w_x = distorted.xy.x - src.x;
                    float w_y = distorted.xy.y - src.y;
                    float w0 = (1.f - w_x) * (1.f - w_y);
                    float w1 = w_x * (1.f - w_y);
                    float w2 = (1.f - w_x) * w_y;
                    float w3 = w_x * w_y;

                    // Fill into lut
                    lut_data[idx].weight[0] = w0;
                    lut_data[idx].weight[1] = w1;
                    lut_data[idx].weight[2] = w2;
                    lut_data[idx].weight[3] = w3;
                }
            }
            else
            {
                lut_data[idx].x = INVALID;
                lut_data[idx].y = INVALID;
            }
        }
    }
}


template<typename T>
void KinectAzure::remap(const k4a_image_t src, const k4a_image_t lut, k4a_image_t dst, interpolation_t type)
{
    int src_width = k4a_image_get_width_pixels(src);
    int dst_width = k4a_image_get_width_pixels(dst);
    int dst_height = k4a_image_get_height_pixels(dst);

    T *src_data = (T *)(void *)k4a_image_get_buffer(src);
    T *dst_data = (T *)(void *)k4a_image_get_buffer(dst);
    coordinate_t *lut_data = (coordinate_t *)(void *)k4a_image_get_buffer(lut);

    memset(dst_data, 0, (size_t)dst_width * (size_t)dst_height * sizeof(T));

    for (int i = 0; i < dst_width * dst_height; i++)
    {
        if (lut_data[i].x != INVALID && lut_data[i].y != INVALID)
        {
            if (type == INTERPOLATION_NEARESTNEIGHBOR)
            {
                dst_data[i] = src_data[lut_data[i].y * src_width + lut_data[i].x];
            }
            else if (type == INTERPOLATION_BILINEAR || type == INTERPOLATION_BILINEAR_DEPTH)
            {
                const T neighbors[4]{ src_data[lut_data[i].y * src_width + lut_data[i].x],
                                             src_data[lut_data[i].y * src_width + lut_data[i].x + 1],
                                             src_data[(lut_data[i].y + 1) * src_width + lut_data[i].x],
                                             src_data[(lut_data[i].y + 1) * src_width + lut_data[i].x + 1] };

                if (type == INTERPOLATION_BILINEAR_DEPTH)
                {
                    // If the image contains invalid data, e.g. depth image contains value 0, ignore the bilinear
                    // interpolation for current target pixel if one of the neighbors contains invalid data to avoid
                    // introduce noise on the edge. If the image is color or ir images, user should use
                    // INTERPOLATION_BILINEAR
                    if (neighbors[0] == 0 || neighbors[1] == 0 || neighbors[2] == 0 || neighbors[3] == 0)
                    {
                        continue;
                    }

                    // Ignore interpolation at large depth discontinuity without disrupting slanted surface
                    // Skip interpolation threshold is estimated based on the following logic:
                    // - angle between two pixels is: theta = 0.234375 degree (120 degree / 512) in binning resolution
                    // mode
                    // - distance between two pixels at same depth approximately is: A ~= sin(theta) * depth
                    // - distance between two pixels at highly slanted surface (e.g. alpha = 85 degree) is: B = A /
                    // cos(alpha)
                    // - skip_interpolation_ratio ~= sin(theta) / cos(alpha)
                    // We use B as the threshold that to skip interpolation if the depth difference in the triangle is
                    // larger than B. This is a conservative threshold to estimate largest distance on a highly slanted
                    // surface at given depth, in reality, given distortion, distance, resolution difference, B can be
                    // smaller
                    const float skip_interpolation_ratio = 0.04693441759f;
                    float depth_min = std::min(std::min(neighbors[0], neighbors[1]),
                                               std::min(neighbors[2], neighbors[3]));
                    float depth_max = std::max(std::max(neighbors[0], neighbors[1]),
                                               std::max(neighbors[2], neighbors[3]));
                    float depth_delta = depth_max - depth_min;
                    float skip_interpolation_threshold = skip_interpolation_ratio * depth_min;
                    if (depth_delta > skip_interpolation_threshold)
                    {
                        continue;
                    }
                }

                dst_data[i] = (T)(neighbors[0] * lut_data[i].weight[0] + neighbors[1] * lut_data[i].weight[1] +
                                         neighbors[2] * lut_data[i].weight[2] + neighbors[3] * lut_data[i].weight[3] +
                                         0.5f);
            }
            else
            {
                printf("Unexpected interpolation type!\n");
                exit(-1);
            }
        }
    }
}

} // namespace fusion


// void KinectAzure::remap(const k4a_image_t src, const k4a_image_t lut, k4a_image_t dst, interpolation_t type)
// {
//     int src_width = k4a_image_get_width_pixels(src);
//     int dst_width = k4a_image_get_width_pixels(dst);
//     int dst_height = k4a_image_get_height_pixels(dst);
//     uint16_t *src_data = (uint16_t *)(void *)k4a_image_get_buffer(src);
//     uint16_t *dst_data = (uint16_t *)(void *)k4a_image_get_buffer(dst);
//     coordinate_t *lut_data = (coordinate_t *)(void *)k4a_image_get_buffer(lut);
//     memset(dst_data, 0, (size_t)dst_width * (size_t)dst_height * sizeof(uint16_t));
//     for (int i = 0; i < dst_width * dst_height; i++)
//     {
//         if (lut_data[i].x != INVALID && lut_data[i].y != INVALID)
//         {
//             // std::cout << " Interpolating " << i << ": " << lut_data[i].x << ", " << lut_data[i].y << std::endl;
//             if (type == INTERPOLATION_NEARESTNEIGHBOR)
//             {
//                 dst_data[i] = src_data[lut_data[i].y * src_width + lut_data[i].x];
//             }
//             else if (type == INTERPOLATION_BILINEAR || type == INTERPOLATION_BILINEAR_DEPTH)
//             {
//                 const uint16_t neighbors[4]{ src_data[lut_data[i].y * src_width + lut_data[i].x],
//                                              src_data[lut_data[i].y * src_width + lut_data[i].x + 1],
//                                              src_data[(lut_data[i].y + 1) * src_width + lut_data[i].x],
//                                              src_data[(lut_data[i].y + 1) * src_width + lut_data[i].x + 1] };
//                 if (type == INTERPOLATION_BILINEAR_DEPTH)
//                 {
//                     // If the image contains invalid data, e.g. depth image contains value 0, ignore the bilinear
//                     // interpolation for current target pixel if one of the neighbors contains invalid data to avoid
//                     // introduce noise on the edge. If the image is color or ir images, user should use
//                     // INTERPOLATION_BILINEAR
//                     if (neighbors[0] == 0 || neighbors[1] == 0 || neighbors[2] == 0 || neighbors[3] == 0)
//                     {
//                         continue;
//                     }
//                     // Ignore interpolation at large depth discontinuity without disrupting slanted surface
//                     // Skip interpolation threshold is estimated based on the following logic:
//                     // - angle between two pixels is: theta = 0.234375 degree (120 degree / 512) in binning resolution
//                     // mode
//                     // - distance between two pixels at same depth approximately is: A ~= sin(theta) * depth
//                     // - distance between two pixels at highly slanted surface (e.g. alpha = 85 degree) is: B = A /
//                     // cos(alpha)
//                     // - skip_interpolation_ratio ~= sin(theta) / cos(alpha)
//                     // We use B as the threshold that to skip interpolation if the depth difference in the triangle is
//                     // larger than B. This is a conservative threshold to estimate largest distance on a highly slanted
//                     // surface at given depth, in reality, given distortion, distance, resolution difference, B can be
//                     // smaller
//                     const float skip_interpolation_ratio = 0.04693441759f;
//                     float depth_min = std::min(std::min(neighbors[0], neighbors[1]),
//                                                std::min(neighbors[2], neighbors[3]));
//                     float depth_max = std::max(std::max(neighbors[0], neighbors[1]),
//                                                std::max(neighbors[2], neighbors[3]));
//                     float depth_delta = depth_max - depth_min;
//                     float skip_interpolation_threshold = skip_interpolation_ratio * depth_min;
//                     if (depth_delta > skip_interpolation_threshold)
//                     {
//                         continue;
//                     }
//                 }
//                 dst_data[i] = (uint16_t)(neighbors[0] * lut_data[i].weight[0] + neighbors[1] * lut_data[i].weight[1] +
//                                          neighbors[2] * lut_data[i].weight[2] + neighbors[3] * lut_data[i].weight[3] +
//                                          0.5f);
//             }
//             else
//             {
//                 printf("Unexpected interpolation type!\n");
//                 exit(-1);
//             }
//         }
//     }
// }

// void KinectAzure::remap_color(const k4a_image_t src, const k4a_image_t lut, k4a_image_t dst, interpolation_t type)
// {
//     int src_width = k4a_image_get_width_pixels(src);
//     int dst_width = k4a_image_get_width_pixels(dst);
//     int dst_height = k4a_image_get_height_pixels(dst);
//     uint32_t *src_data = (uint32_t *)(void *)k4a_image_get_buffer(src);
//     std::cout << src_data[0] << std::endl;
//     uint32_t *dst_data = (uint32_t *)(void *)k4a_image_get_buffer(dst);
//     coordinate_t *lut_data = (coordinate_t *)(void *)k4a_image_get_buffer(lut);
//     memset(dst_data, 0, (size_t)dst_width * (size_t)dst_height * sizeof(uint32_t));
//     for (int i = 0; i < dst_width * dst_height; i++)
//     {
//         if (lut_data[i].x != INVALID && lut_data[i].y != INVALID)
//         {
//             std::cout << " Interpolating " << i << ": " << lut_data[i].x << ", " << lut_data[i].y << std::endl;
//             if (type == INTERPOLATION_NEARESTNEIGHBOR)
//             {
//                 dst_data[i] = src_data[lut_data[i].y * src_width + lut_data[i].x];
//             }
//             else if (type == INTERPOLATION_BILINEAR || type == INTERPOLATION_BILINEAR_DEPTH)
//             {
//                 const uint32_t neighbors[4]{ src_data[lut_data[i].y * src_width + lut_data[i].x],
//                                              src_data[lut_data[i].y * src_width + lut_data[i].x + 1],
//                                              src_data[(lut_data[i].y + 1) * src_width + lut_data[i].x],
//                                              src_data[(lut_data[i].y + 1) * src_width + lut_data[i].x + 1] };
//                 if (type == INTERPOLATION_BILINEAR_DEPTH)
//                 {
//                     // If the image contains invalid data, e.g. depth image contains value 0, ignore the bilinear
//                     // interpolation for current target pixel if one of the neighbors contains invalid data to avoid
//                     // introduce noise on the edge. If the image is color or ir images, user should use
//                     // INTERPOLATION_BILINEAR
//                     if (neighbors[0] == 0 || neighbors[1] == 0 || neighbors[2] == 0 || neighbors[3] == 0)
//                     {
//                         continue;
//                     }
//                     // Ignore interpolation at large depth discontinuity without disrupting slanted surface
//                     // Skip interpolation threshold is estimated based on the following logic:
//                     // - angle between two pixels is: theta = 0.234375 degree (120 degree / 512) in binning resolution
//                     // mode
//                     // - distance between two pixels at same depth approximately is: A ~= sin(theta) * depth
//                     // - distance between two pixels at highly slanted surface (e.g. alpha = 85 degree) is: B = A /
//                     // cos(alpha)
//                     // - skip_interpolation_ratio ~= sin(theta) / cos(alpha)
//                     // We use B as the threshold that to skip interpolation if the depth difference in the triangle is
//                     // larger than B. This is a conservative threshold to estimate largest distance on a highly slanted
//                     // surface at given depth, in reality, given distortion, distance, resolution difference, B can be
//                     // smaller
//                     const float skip_interpolation_ratio = 0.04693441759f;
//                     float depth_min = std::min(std::min(neighbors[0], neighbors[1]),
//                                                std::min(neighbors[2], neighbors[3]));
//                     float depth_max = std::max(std::max(neighbors[0], neighbors[1]),
//                                                std::max(neighbors[2], neighbors[3]));
//                     float depth_delta = depth_max - depth_min;
//                     float skip_interpolation_threshold = skip_interpolation_ratio * depth_min;
//                     if (depth_delta > skip_interpolation_threshold)
//                     {
//                         continue;
//                     }
//                 }
//                 dst_data[i] = (uint32_t)(neighbors[0] * lut_data[i].weight[0] + neighbors[1] * lut_data[i].weight[1] +
//                                          neighbors[2] * lut_data[i].weight[2] + neighbors[3] * lut_data[i].weight[3] +
//                                          0.5f);
//             }
//             else
//             {
//                 printf("Unexpected interpolation type!\n");
//                 exit(-1);
//             }
//         }
//     }
// }