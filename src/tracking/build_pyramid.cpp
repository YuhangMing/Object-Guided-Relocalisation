#include "tracking/build_pyramid.h"

namespace fusion
{

FUSION_HOST void build_intensity_pyr(cv::cuda::GpuMat base, std::vector<cv::cuda::GpuMat> &pyr, const int level)
{
    if (pyr.size() != level)
        pyr.resize(level);

    pyr[0] = base;
    for (int i = 1; i < level; ++i)
    {
        pyrDownImage(pyr[i - 1], pyr[i]);
    }
}

FUSION_HOST void build_intensity_dxdy_pyr(std::vector<cv::cuda::GpuMat> intensity_pyr, 
                                          std::vector<cv::cuda::GpuMat> &dx_pyr, 
                                          std::vector<cv::cuda::GpuMat> &dy_pyr)
{
    if (dx_pyr.size() != intensity_pyr.size())
        dx_pyr.resize(intensity_pyr.size());

    if (dy_pyr.size() != intensity_pyr.size())
        dy_pyr.resize(intensity_pyr.size());

    for (int i = 0; i < intensity_pyr.size(); ++i)
    {
        computeDerivative(intensity_pyr[i], dx_pyr[i], dy_pyr[i]);
    }
}

FUSION_HOST void build_depth_pyr(cv::cuda::GpuMat base, std::vector<cv::cuda::GpuMat> &pyr, const int level)
{
    if (pyr.size() != level)
        pyr.resize(level);

    filterDepthBilateral(base, pyr[0]);
    for (int i = 1; i < level; ++i)
    {
        pyrDownDepth(pyr[i - 1], pyr[i]);
    }
}

FUSION_HOST void build_vnmap_pyr(std::vector<cv::cuda::GpuMat> depth_pyr, 
                                 std::vector<cv::cuda::GpuMat> &vmap_pyr, 
                                 std::vector<cv::cuda::GpuMat> &nmap_pyr,
                                 const std::vector<Eigen::Matrix3f> KInv_pyr)
{
    if (vmap_pyr.size() != depth_pyr.size())
        vmap_pyr.resize(depth_pyr.size());

    if (nmap_pyr.size() != depth_pyr.size())
        nmap_pyr.resize(depth_pyr.size());

    for (int i = 0; i < depth_pyr.size(); ++i)
    {
        backProjectDepth(depth_pyr[i], vmap_pyr[i], KInv_pyr[i]);
        computeNMap(vmap_pyr[i], nmap_pyr[i]);
    }
}


FUSION_HOST void build_vmap_pyr(std::vector<cv::cuda::GpuMat> depth_pyr, 
                                std::vector<cv::cuda::GpuMat> vmap_pyr, 
                                const std::vector<Eigen::Matrix3f> KInv_pyr)
{
    if (vmap_pyr.size() != depth_pyr.size())
        vmap_pyr.resize(depth_pyr.size());

    for (int i = 0; i < depth_pyr.size(); ++i)
    {
        backProjectDepth(depth_pyr[i], vmap_pyr[i], KInv_pyr[i]);
    }
}

FUSION_HOST void build_vmap_pyr(cv::cuda::GpuMat base, std::vector<cv::cuda::GpuMat> &vmap_pyr, const int level)
{
    if (vmap_pyr.size() != level)
        vmap_pyr.resize(level);

    vmap_pyr[0] = base;
    for (int i = 1; i < vmap_pyr.size(); ++i)
    {
        pyrDownVMap(vmap_pyr[i - 1], vmap_pyr[i]);
    }
}

FUSION_HOST void build_nmap_pyr(std::vector<cv::cuda::GpuMat> vmap_pyr, 
                                std::vector<cv::cuda::GpuMat> &nmap_pyr)
{
    if (nmap_pyr.size() != vmap_pyr.size())
        nmap_pyr.resize(vmap_pyr.size());

    for (int i = 0; i < vmap_pyr.size(); ++i)
    {
        computeNMap(vmap_pyr[i], nmap_pyr[i]);
    }
}

} // namespace fusion