#include "utils/settings.h"
#include <cmath>

void SetCalibration()
{
    GlobalCfg.mK(0, 0) = GlobalCfg.fx;
    GlobalCfg.mK(1, 1) = GlobalCfg.fy;
    GlobalCfg.mK(0, 2) = GlobalCfg.cx;
    GlobalCfg.mK(1, 2) = GlobalCfg.cy;
    GlobalCfg.invfx = 1.0 / GlobalCfg.fx;
    GlobalCfg.invfy = 1.0 / GlobalCfg.fy;

    GlobalCfg.mKInv = GlobalCfg.mK.inverse();

    GlobalCfg.mCvK = cv::Mat::eye(3, 3, CV_32F);
    GlobalCfg.mCvK.at<float>(0, 0) = GlobalCfg.fx;
    GlobalCfg.mCvK.at<float>(1, 1) = GlobalCfg.fy;
    GlobalCfg.mCvK.at<float>(0, 2) = GlobalCfg.cx;
    GlobalCfg.mCvK.at<float>(1, 2) = GlobalCfg.cy;

    // GlobalCfg.mCvDistCoeff.create(4, 1, CV_32FC1);
    // GlobalCfg.mCvDistCoeff.at<float>(0) = GlobalCfg.mDistCoeff[0];
    // GlobalCfg.mCvDistCoeff.at<float>(1) = GlobalCfg.mDistCoeff[1];
    // GlobalCfg.mCvDistCoeff.at<float>(2) = GlobalCfg.mDistCoeff[2];
    // GlobalCfg.mCvDistCoeff.at<float>(3) = GlobalCfg.mDistCoeff[3];
    // if (GlobalCfg.mDistCoeff[4] != 0)
    // {
    //     GlobalCfg.mCvDistCoeff.resize(5);
    //     GlobalCfg.mCvDistCoeff.at<float>(4) = GlobalCfg.mDistCoeff[4];
    // }

    // if (GlobalCfg.mCvDistCoeff.at<float>(0) != 0.0)
    // {
    //     cv::Mat mat(4, 2, CV_32F);
    //     mat.at<float>(0, 0) = 0.0;
    //     mat.at<float>(0, 1) = 0.0;
    //     mat.at<float>(1, 0) = GlobalCfg.mWidth;
    //     mat.at<float>(1, 1) = 0.0;
    //     mat.at<float>(2, 0) = 0.0;
    //     mat.at<float>(2, 1) = GlobalCfg.mHeight;
    //     mat.at<float>(3, 0) = GlobalCfg.mWidth;
    //     mat.at<float>(3, 1) = GlobalCfg.mHeight;

    //     // Undistort corners
    //     mat = mat.reshape(2);
    //     cv::undistortPoints(
    //         mat, mat, GlobalCfg.mCvK, GlobalCfg.mCvDistCoeff,
    //         cv::Mat(), GlobalCfg.mCvK);
    //     mat = mat.reshape(1);

    //     GlobalCfg.mMinImageWidth = std::min(mat.at<float>(0, 0), mat.at<float>(2, 0));
    //     GlobalCfg.mMaxImageWidth = std::max(mat.at<float>(1, 0), mat.at<float>(3, 0));
    //     GlobalCfg.mMinImageHeight = std::min(mat.at<float>(0, 1), mat.at<float>(1, 1));
    //     GlobalCfg.mMaxImageHeight = std::max(mat.at<float>(2, 1), mat.at<float>(3, 1));
    // }
    // else
    // {
    //     GlobalCfg.mMinImageWidth = 0.0f;
    //     GlobalCfg.mMaxImageWidth = GlobalCfg.mWidth;
    //     GlobalCfg.mMinImageHeight = 0.0f;
    //     GlobalCfg.mMaxImageHeight = GlobalCfg.mHeight;
    // }

    // GlobalCfg.mFeatureGridWidthInv = static_cast<float>(FEATURE_GRID_COLS) / static_cast<float>(GlobalCfg.mMaxImageWidth - GlobalCfg.mMinImageWidth);
    // GlobalCfg.mFeatureGraidHeightInv = static_cast<float>(FEATURE_GRID_ROWS) / static_cast<float>(GlobalCfg.mMaxImageHeight - GlobalCfg.mMinImageHeight);
    GlobalCfg.mfDepthTh = GlobalCfg.mfBaseline * GlobalCfg.mfDepthTh / GlobalCfg.fx;
}

bool CheckImageBound(const Eigen::Vector3f &pt)
{
    const float &fx = GlobalCfg.fx;
    const float &fy = GlobalCfg.fy;
    const float &cx = GlobalCfg.cx;
    const float &cy = GlobalCfg.cy;
    const float u = fx * pt[0] / pt[2] + cx;
    const float v = fy * pt[1] / pt[2] + cy;
    return CheckImageBound(u, v);
}

bool CheckImageBound(const float &u, const float &v)
{
    if (u >= GlobalCfg.mMinImageWidth && v >= GlobalCfg.mMinImageHeight &&
        u <= GlobalCfg.mMaxImageWidth - 1 && v <= GlobalCfg.mMaxImageHeight - 1)
        return true;
    return false;
}

Config GlobalCfg;
