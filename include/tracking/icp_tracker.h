#ifndef FUSION_ICP_TRACKER_H
#define FUSION_ICP_TRACKER_H

#include <memory>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>
#include "data_struct/rgbd_frame.h"
#include "tracking/device_image.h"

namespace fusion
{

struct TrackingResult
{
  bool sucess;
  float icp_error;
  Sophus::SE3d update;
};

struct TrackingContext
{
  bool use_initial_guess_;
  std::vector<Eigen::Matrix3f> K_pyr_;
  std::vector<int> max_iterations_;
  Sophus::SE3d initial_estimate_;
};

class DenseTracking
{
public:
  DenseTracking();
  // DenseTracking(const IntrinsicMatrix K, const int NUM_PYR);
  TrackingResult compute_transform(const RgbdImagePtr reference, const RgbdImagePtr current, const TrackingContext &c);
  // TrackingResult compute_transform_depth_only(const RgbdImagePtr reference, const RgbdImagePtr current, const TrackingContext &c);
  // TrackingResult compute_transform_depth_centroids(const RgbdImagePtr reference, const RgbdImagePtr current, const TrackingContext &c);

  TrackingResult compute_transform(const TrackingContext &c);
  void swap_intensity_pyr();
  void set_source_image(cv::Mat img, const int num_pyr);
  void set_source_intensity(cv::cuda::GpuMat intensity, const int num_pyr);
  void set_source_depth(cv::Mat depth, const std::vector<Eigen::Matrix3f> KInv_pyr);
  // void set_source_vmap(cv::cuda::GpuMat vmap);
  void set_reference_image(cv::Mat img, const int num_pyr);
  void set_reference_intensity(cv::cuda::GpuMat intensity, const int num_pyr);
  void set_reference_depth(cv::Mat depth, const std::vector<Eigen::Matrix3f> KInv_pyr);
  // void set_reference_vmap(cv::cuda::GpuMat vmap);

  cv::cuda::GpuMat get_vmap_src(const int &level = 0);
  cv::cuda::GpuMat get_nmap_src(const int &level = 0);
  cv::cuda::GpuMat get_depth_src(const int &level = 0);
  cv::cuda::GpuMat get_intensity_src(const int &level = 0);
  cv::cuda::GpuMat get_intensity_dx(const int &level = 0);
  cv::cuda::GpuMat get_intensity_dy(const int &level = 0);
  cv::cuda::GpuMat get_vmap_ref(const int &level = 0);
  cv::cuda::GpuMat get_nmap_ref(const int &level = 0);
  cv::cuda::GpuMat get_intensity_ref(const int &level = 0);

private:
  std::vector<cv::cuda::GpuMat> intensity_src_pyr;
  std::vector<cv::cuda::GpuMat> intensity_dx_pyr;
  std::vector<cv::cuda::GpuMat> intensity_dy_pyr;
  std::vector<cv::cuda::GpuMat> depth_src_pyr;
  std::vector<cv::cuda::GpuMat> vmap_src_pyr;
  std::vector<cv::cuda::GpuMat> nmap_src_pyr;

  std::vector<cv::cuda::GpuMat> intensity_ref_pyr;
  std::vector<cv::cuda::GpuMat> depth_ref_pyr;
  std::vector<cv::cuda::GpuMat> vmap_ref_pyr;
  std::vector<cv::cuda::GpuMat> nmap_ref_pyr;

  Eigen::Matrix<float, 6, 6> icp_hessian;
  Eigen::Matrix<float, 6, 6> rgb_hessian;
  Eigen::Matrix<float, 6, 6> joint_hessian;

  Eigen::Matrix<float, 6, 1> icp_residual;
  Eigen::Matrix<float, 6, 1> rgb_residual;
  Eigen::Matrix<float, 6, 1> joint_residual;
  Eigen::Matrix<double, 6, 1> update;

  Eigen::Matrix<float, 2, 1> residual_icp_;
  Eigen::Matrix<float, 2, 1> residual_rgb_;

  cv::cuda::GpuMat SUM_SE3;
  cv::cuda::GpuMat OUT_SE3;

  float last_icp_error;
  float last_rgb_error;

  std::vector<int> max_iterations;

  // Point-to-Point ICP Probabilistic model
  // cent icp step and returns hessian
  Eigen::Matrix<float, 6, 6> cent_hessian;
  Eigen::Matrix<float, 6, 1> cent_residual;
  Eigen::Matrix<float, 2, 1> residual_cent_;
  float last_cent_error;
  void cent_reduce(
      const cv::Mat &curr_cent,
      const cv::Mat &last_cent,
      const Sophus::SE3d &pose,
      float *jtj, float *jtr,
      float *residual);
};

} // namespace fusion

#endif