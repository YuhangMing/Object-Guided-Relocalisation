#ifndef RAY_TRACE_ENGINE
#define RAY_TRACE_ENGINE

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>
#include "mapping/VoxelMap.h"

class MapStruct;

class RayTraceEngine
{
public:
    ~RayTraceEngine();
    RayTraceEngine(int w, int h, const Eigen::Matrix3f &K);
    void RayTrace(MapStruct *pMapStruct, const Sophus::SE3d &Tcw);
    struct RenderingBlock
    {
        Eigen::Matrix<short, 2, 1> upper_left;
        Eigen::Matrix<short, 2, 1> lower_right;
        Eigen::Vector2f zrange;
    };

    uint GetNumVisibleBlock();
    uint GetNumRenderingBlocks();
    cv::cuda::GpuMat GetVMap();

protected:
    void UpdateRenderingBlocks(MapStruct *pMS, const Sophus::SE3d &Tcw);
    void reset();
    float fx, fy, cx, cy, invfx, invfy;
    Eigen::Matrix3f mK;
    cv::cuda::GpuMat mTracedvmap;
    cv::cuda::GpuMat mTracedImage;
    cv::cuda::GpuMat mDepthMapMin;
    cv::cuda::GpuMat mDepthMapMax;
    uint *mpNumVisibleBlocks;
    uint *mpNumRenderingBlocks;
    RenderingBlock *mplRenderingBlockList;

    int w, h;
};

#endif