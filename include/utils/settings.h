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

// Debug
struct Config
{
    bool bEnableViewer = true;
    bool bPauseWindow = false;
    bool bSemantic = true;
    bool bSubmapping = false;
    bool bPureReloc = false;
    // std::string data_path = "/home/yohann/SLAMs/datasets/BOR/";
    std::string data_path = "/home/yohann/SLAMs/datasets/OS/";
    
    bool bLoadDiskMap = false;
    bool bVisMapReg = false;    // display two color for pair-wise map registration
    // set to 1 for relocalisation tests
    // set to 4 for visualising maps registration
    int mapSize = 1;    
    std::string map_file = "map";
    
    bool bOutputPose = true;
    std::string output_pose_file = "/home/yohann/SLAMs/Object-Guided-Relocalisation/pose_info/";
    
    bool bRecord = false;
    std::string record_dir = "/home/yohann/SLAMs/datasets/sequence/";

    int width = 640;
    int height = 480;
    float depthScale = 1000.0;
    // // Asus
    // float fx = 580;
    // float fy = 580;
    // float cx = 319.5;
    // float cy = 239.5;
    // Azure
    float fx = 607.665;
    float fy = 607.516;
    float cx = 321.239;
    float cy = 245.043;
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
    
    int mMinImageWidth = 0;
    int mMaxImageWidth = 640;
    int mMinImageHeight = 0;
    int mMaxImageHeight = 480;

    int maxPyramidLevel = 5;
    // // ToDo: add more tracking options
    // bool mbUseRGB = true;
    // bool mbUseDepth = true;
    // float mfMaxDepth = 5.0;
    // float mfRGBGradTh = 16;
    // bool mbFrameToMap = true;
    // float mfICPWeight = 0.999;

    // // ToDo: add more system control options
    // int mCurrentFrameId = 0;
    // bool mbEnableDebugLogs = false;
    // bool mbEnableDebugPlot = false;
    // bool mbUseGroundTruth = false;
    // bool mbEmulateRealTime = false;
    // int mbEmulateFrameRate = 30;
    // bool mbIsPaused = false;
    // bool mbStopRequested = false;
    // bool mbStopWhenEnd = false;
};

extern Config GlobalCfg;
void SetCalibration();
bool CheckImageBound(const Eigen::Vector3f &pt);
bool CheckImageBound(const float &u, const float &v);