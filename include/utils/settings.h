#pragma once
#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>

// #define FEATURE_GRID_COLS 64
// #define FEATURE_GRID_ROWS 48

// #define PCD_POINT_IN_NUMBER 8192
// #define PCD_POINT_DIMENSION 6
// #define PCD_LOCAL_FEATURE_DIM 128

// #define MIN_KEYFRAME_NEW_LOOP 5
#define MAX_RANSAC_ITERAITON_LOOP 1000

// #define NO_RET(x) \
//     if ((x))      \
//         ;

typedef Sophus::SE3d PoseType;
// typedef Eigen::Vector<float, 132> LocalFeatType;
// typedef Eigen::Vector<float, 256> GlobalFeatType;

// Debug
struct Config
{
    bool bSubmapping = false;
    bool bSemantic = false;
    bool bRecord = false;
    bool bEnableViewer = true;
    bool bLoadDiskMap = false;
    
    std::string data_folder = "/home/yohann/SLAMs/datasets/BOR/";
    // consider directly load all filenames in the directly
    // and remove the dependency on this array
    int num_img[10] = {3522, 1237, 887, 1221, 809, 1141, 919, 1470, 501, 870};
    std::string map_file = "map";
    int mapSize = 4;

    int mCurrentFrameId = 0;
    bool mbEnableDebugLogs = false;
    bool mbEnableDebugPlot = false;
    std::string mStrOutputPath = "";
    bool mbUseGroundTruth = false;
    bool mbEmulateRealTime = false;
    int mbEmulateFrameRate = 30;
    bool mbIsPaused = false;
    bool mbStopRequested = false;
    bool mbStopWhenEnd = false;

    int width = 640;
    int height = 480;
    float fx = 580;
    float fy = 580;
    float cx = 319.5;
    float cy = 239.5;
    float invfx;
    float invfy;
    Eigen::Matrix3f K;
    Eigen::Matrix3f KInv;
    cv::Mat cvK;
    // Eigen::Vector<float, 5> mDistCoeff;
    // cv::Mat mCvDistCoeff;
    float mfBaseline = 40;
    float mfDepthTh = 40;
    PoseType mInitPose;

    int maxPyramidLevel = 5;
    bool mbUseRGB = true;
    bool mbUseDepth = true;
    float mfMaxDepth = 5.0;
    float mfDepthScale = 1000.0;
    float mfRGBGradTh = 16;
    bool mbFrameToMap = true;
    float mfICPWeight = 0.999;
    bool mbEnableLocalBA = true;
    bool mbTrackLocalMap = true;
    
    // int mNFeatures = 500;
    // int mNOctaves = 8;
    // int mFastInitTh = 20;
    // int mFastMinTh = 7;
    // float mfScaleFactor = 1.2;

    int mMinImageWidth = 0;
    int mMaxImageWidth = 640;
    int mMinImageHeight = 0;
    int mMaxImageHeight = 480;
    // int mFeatureGridWidthInv;
    // int mFeatureGraidHeightInv;
};

extern Config GlobalCfg;
void SetCalibration();
bool CheckImageBound(const Eigen::Vector3f &pt);
bool CheckImageBound(const float &u, const float &v);