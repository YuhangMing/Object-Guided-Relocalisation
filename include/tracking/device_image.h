#ifndef DEVICE_IMAGE_H
#define DEVICE_IMAGE_H

#include <memory>
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>
// #include <opencv2/improc.hpp>
#include <opencv2/cudaarithm.hpp>
#include "data_struct/rgbd_frame.h"

namespace fusion
{

class DeviceImage;
typedef std::shared_ptr<DeviceImage> RgbdImagePtr;

class DeviceImage
{
public:
    DeviceImage() = default;
    DeviceImage(const std::vector<Eigen::Matrix3f> vIntrinsicsInv);
    DeviceImage(const DeviceImage &);
    DeviceImage &operator=(DeviceImage);
    friend void swap(DeviceImage &first, DeviceImage &second);
    friend void copyDeviceImage(RgbdImagePtr src, RgbdImagePtr dst);

    void resize_pyramid(const int &max_level);
    void resize_device_map();
    void upload(const std::shared_ptr<RgbdFrame> frame);
    void downloadVNM(RgbdFramePtr frame, bool bTrackLost);
    
    RgbdFramePtr get_reference_frame() const;
    cv::cuda::GpuMat get_rendered_image();
    cv::cuda::GpuMat get_rendered_scene_textured();
    cv::cuda::GpuMat get_depth(const int &level = 0) const;
    cv::cuda::GpuMat get_raw_depth() const;
    cv::cuda::GpuMat get_image() const;
    cv::cuda::GpuMat &get_vmap(const int &level = 0);
    cv::cuda::GpuMat &get_nmap(const int &level = 0);
    cv::cuda::GpuMat get_intensity(const int &level = 0) const;
    cv::cuda::GpuMat get_intensity_dx(const int &level = 0) const;
    cv::cuda::GpuMat get_intensity_dy(const int &level = 0) const;

    cv::Mat get_centroids() const;

    // TODO: move functions here
    void create_depth_pyramid(const int max_level, const bool use_filter = true);
    void create_intensity_pyramid(const int max_level);
    void create_vmap_pyramid(const int max_level); // TODO
    void create_nmap_pyramid(const int max_level); // TODO

    // int NUM_PYRS;

    // data structure for maskRCNN detection results
    void upload_semantics(const std::shared_ptr<RgbdFrame> frame);
    void GeometricRefinement(float lamb, float tao, int win_size, cv::Mat &edge_device);
    cv::cuda::GpuMat &get_object_mask();
    cv::cuda::GpuMat get_object_bbox(const int &i) const;
    int get_object_label(const int &i) const;

    // !!!!! COMMENT this line out later
    cv::cuda::GpuMat get_rendered_image_raw();


    std::vector<Eigen::Matrix3f> vKInv;
private:
    RgbdFramePtr reference_frame;

    // original image in CV_8UC3
    cv::cuda::GpuMat image;

    // original image in CV_32FC3
    // this is needed when converting to grayscale
    // otherwise lose accuracy due to tuncation error
    cv::cuda::GpuMat image_float;

    // original depth in CV_32FC1
    cv::cuda::GpuMat depth_float;

    // base intensity in CV_32FC1
    cv::cuda::GpuMat intensity_float;

    // for tracking
    std::vector<cv::cuda::GpuMat> depth_pyr;        // CV_32FC1
    std::vector<cv::cuda::GpuMat> intensity_pyr;    // CV_32FC1
    std::vector<cv::cuda::GpuMat> intensity_dx_pyr; // CV_32FC1
    std::vector<cv::cuda::GpuMat> intensity_dy_pyr; // CV_32FC1
    std::vector<cv::cuda::GpuMat> vmap_pyr;         // CV_32FC4
    std::vector<cv::cuda::GpuMat> nmap_pyr;         // CV_32FC4

    // for relocalisation
    // cv::cuda::GpuMat cent_float;    // CV_32FC1, rows=6; cols=3;
    cv::Mat cent_float_cpu;

    // for debugging and visualization
    std::vector<cv::cuda::GpuMat> semi_dense_image;
    cv::cuda::GpuMat rendered_image;
    cv::cuda::GpuMat rendered_image_textured;
    cv::cuda::GpuMat rendered_image_raw;

    // std::vector<fusion::IntrinsicMatrix> cam_params;

    // data structure for maskRCNN detection results
    cv::cuda::GpuMat detected_masks;    // DeviceArray2D<unsigned char> detected_masks;
    std::vector<cv::cuda::GpuMat> v_detected_bboxes;
    // cv::cuda::GpuMat detected_bboxes;
};

} // namespace fusion

#endif