#pragma once

#include <k4a/k4a.h>
// #include <k4arecord/record.h>
// #include <k4arecord/playback.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#define INVALID INT32_MIN

namespace fusion
{

typedef struct _pinhole_t
{
    float px;
    float py;
    float fx;
    float fy;

    int width;
    int height;
} pinhole_t;

typedef struct _coordinate_t
{
    int x;
    int y;
    float weight[4];
} coordinate_t;

typedef enum
{
    INTERPOLATION_NEARESTNEIGHBOR, /**< Nearest neighbor interpolation */
    INTERPOLATION_BILINEAR,        /**< Bilinear interpolation */
    INTERPOLATION_BILINEAR_DEPTH   /**< Bilinear interpolation with invalidation when neighbor contain invalid
                                                 data with value 0 */
} interpolation_t;

class KinectAzure
{
public:
    KinectAzure();
    ~KinectAzure();
    bool getNextPair(cv::Mat &rgb, cv::Mat &depth);

private:
    int frameIdx = 0;

    k4a_device_t device = NULL;
    k4a_capture_t capture = NULL;
    k4a_calibration_t calibration;
    k4a_transformation_t transformation;

    k4a_image_t color_image = NULL; 
    k4a_image_t depth_image = NULL;
    // k4a_image_t transformed_color_image = NULL; 
    k4a_image_t transformed_depth_image = NULL;;
    k4a_image_t lut = NULL;
    k4a_image_t undistorted_color_image = NULL;
    k4a_image_t undistorted_depth_image = NULL;
    
    pinhole_t pinhole;

    void k4a_image_to_cv_mat(const k4a_image_t kImg, cv::Mat &mImg, int typeFlag);
    void print_k4a_image_info(const k4a_image_t kImg, std::string str);

    /* 
    Compute color point cloud by warping depth image into color camera geometry with downscaled color image and
    downscaled calibration. 
    This example's goal is to show how to configure the calibration and use the transformation API as it is when
    the user does not need a point cloud from high resolution transformed depth image. 
    The downscaling method here is naively to average binning 2x2 pixels, user should choose their own appropriate 
    downscale method on the color image, this example is only demonstrating the idea. 
    However, no matter what scale you choose to downscale the color image, please keep the aspect ratio unchanged
    (to ensure thedistortion parameters from original calibration can still be used for the downscaled image).
    */
    k4a_image_t downscale_image_2x2_binning(const k4a_image_t color_image);

    static void compute_xy_range(const k4a_calibration_t *calibration, const k4a_calibration_type_t camera,
                                 const int width, const int height, float &x_min, float &x_max, 
                                 float &y_min, float &y_max);
    static pinhole_t create_pinhole_from_xy_range(const k4a_calibration_t *calibration, 
                                                  const k4a_calibration_type_t camera);
    static void create_undistortion_lut(const k4a_calibration_t *calibration, const k4a_calibration_type_t camera,
                                        const pinhole_t *pinhole, k4a_image_t lut, interpolation_t type);
    
    template<typename T>
    static void remap(const k4a_image_t src, const k4a_image_t lut, k4a_image_t dst, interpolation_t type);
    
};

} // namespace fusion