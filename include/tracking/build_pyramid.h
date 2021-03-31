#ifndef FUSION_BUILD_PYRAMID_H
#define FUSION_BUILD_PYRAMID_H

#include "utils/macros.h"
#include "tracking/cuda_imgproc.h"

namespace fusion
{

FUSION_HOST void build_intensity_pyr(cv::cuda::GpuMat base, std::vector<cv::cuda::GpuMat> &pyr, const int level);
FUSION_HOST void build_intensity_dxdy_pyr(std::vector<cv::cuda::GpuMat> intensity_pyr, 
                                          std::vector<cv::cuda::GpuMat> &dx_pyr, 
                                          std::vector<cv::cuda::GpuMat> &dy_pyr);
FUSION_HOST void build_depth_pyr(cv::cuda::GpuMat base, std::vector<cv::cuda::GpuMat> &pyr, const int level);
FUSION_HOST void build_vnmap_pyr(std::vector<cv::cuda::GpuMat> depth_pyr, 
                                 std::vector<cv::cuda::GpuMat> &vmap_pyr, 
                                 std::vector<cv::cuda::GpuMat> &nmap_pyr,
                                 const std::vector<Eigen::Matrix3f> KInv_pyr);
FUSION_HOST void build_vmap_pyr(std::vector<cv::cuda::GpuMat> depth_pyr, 
                                std::vector<cv::cuda::GpuMat> &vmap_pyr, 
                                const std::vector<Eigen::Matrix3f> KInv_pyr);
FUSION_HOST void build_vmap_pyr(cv::cuda::GpuMat base, std::vector<cv::cuda::GpuMat> &vmap_pyr, const int level);
FUSION_HOST void build_nmap_pyr(std::vector<cv::cuda::GpuMat> vmap_pyr, 
                                std::vector<cv::cuda::GpuMat> &nmap_pyr);

} // namespace fusion

#endif