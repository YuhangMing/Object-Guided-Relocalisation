#pragma once

#include <mutex>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <pangolin/pangolin.h>
#include <pangolin/gl/glcuda.h>
#include "mapping/VoxelMap.h"

namespace slam
{
    class FeatureMap;
    class MapDrawer
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        MapDrawer(FeatureMap *pMap);
        void DrawMapPoints(int iPointSize);
        void DrawKeyframes(bool bDrawKF, bool bDrawGraph, int iEdgeWeight);
        void DrawMesh(int N, const pangolin::OpenGlMatrix &mvpMat);
        void LinkGlSlProgram();
        // std::vector<KeyFrame *> GetKeyframesAll();

    private:
        FeatureMap *mpMap;
        std::mutex pose_mutex;
        Eigen::Matrix4f mCameraPose;
        Eigen::Matrix3f mCalibInv;
        int width, height;
        pangolin::GlSlProgram mShader;
    };

} // namespace slam
