#ifndef FUSION_RGBD_FRAME_H
#define FUSION_RGBD_FRAME_H

#include <map>
#include <mutex>
#include <memory>
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include "data_struct/map_point.h"
#include "data_struct/map_cuboid.h"
#include "data_struct/map_object.h"
#include "detection/detector.h"

namespace fusion
{

class RgbdFrame;
using RgbdFramePtr = std::shared_ptr<RgbdFrame>;

class RgbdFrame
{
public:
  RgbdFrame(const cv::Mat &depth, const cv::Mat &image, const size_t id, const double ts);
  RgbdFrame();
  void copyTo(RgbdFramePtr dst);

  std::vector<cv::KeyPoint> cv_key_points;
  std::vector<std::shared_ptr<Point3d>> key_points;
  std::map<RgbdFramePtr, Eigen::Matrix4f> neighbours;
  cv::Mat descriptors;

  std::size_t id;
  double timeStamp;
  Sophus::SE3d pose;

  cv::Mat image;
  // cv::Mat img_original;
  cv::Mat depth;
  // vmap in CV_32FC4
  cv::Mat vmap;     // vmap & nmap are calculated in uploading process
  cv::Mat nmap;     // and then are downloaded back here, IN WORLD Coordinate System
  // cv::Mat image_backup;
  // cv::Mat world_plane;
  int row_frame, col_frame;

  // data structure for maskRCNN/NOCS detection results
  int numDetection;
  std::vector<cv::Mat> vMasks;
  std::vector<int> vLabels;
  std::vector<float> vScores;
  std::vector<cv::Mat> vCoords;
  // std::vector<std::shared_ptr<Cuboid3d>> vCuboids;
  std::vector<std::shared_ptr<Object3d>> vObjects;
  cv::Mat cent_matrix;
  cv::Mat mask, nocs_map;   // combined masks with values = labels
  // std::vector<cv::Mat> v_bbox;
  
  // Geometric Segmentation results
  cv::Mat mEdge;
  int nConComps;
  cv::Mat mLabeled, mStats, mCentroids;
  int palette[7][3] = {
        {0,   0,   0},
        {255, 165, 0},
        {128, 128, 0},
        {128, 0,   128},
        {255, 255, 0},
        {0,   0,   255},
        {0,   255, 255}
    };
  // Plane information
  std::vector<Eigen::Vector3f> plane_normals;

  // semantic related functions
  void ExtractSemantics(semantic::Detector* detector, bool bBbox, bool bContour, bool bText);
  // object
  void ExtractObjects(semantic::Detector* detector, bool bBbox, bool bContour, bool bText);
  // void GeometricRefinement(float lamb, float tao, int win_size);
  void FuseMasks(cv::Mat edge, int thre);
  cv::Scalar CalculateColor(long int label);
  cv::Mat Array2Mat(int* aMask);
  cv::Mat Array2Mat(float* aMask);
  // cv::Mat Array2Mat(float* aObj, int num);
  // plane
  void ExtractPlanes();

  void UpdateCentMatrix(std::vector<std::shared_ptr<Object3d>> map_obj,
                        std::vector<std::pair<int, int>> v_best_map_cub_labidx);
  // void ReprojectMapInliers(std::vector<std::shared_ptr<Object3d>> map_obj,
  //                       std::vector<std::pair<int, int>> v_inlier_pairs,
  //                       std::vector<std::pair<int, int>> v_best_map_cub_labidx);
  // void UpdateFrameInliers(std::vector<std::pair<int, int>> v_inlier_pairs,
  //                         std::vector<std::pair<int, int>> v_best_map_cub_labidx);
  void ReprojectMapInliers(std::vector<std::shared_ptr<Object3d>> map_obj,
                        std::vector<std::pair<int, std::pair<int, int>>> v_inlier_pairs);
  void UpdateFrameInliers(std::vector<std::pair<int, std::pair<int, int>>> v_inlier_pairs);

private:
  std::vector<Eigen::Vector3f> FitPose(cv::Mat coord, cv::Mat mask, Eigen::Matrix4f& Twn, std::vector<float>& dims, float& scale, Eigen::Matrix3d& Sigma_t);
  Eigen::Matrix4f EstimateSimilarityTransform(std::vector<Eigen::Vector3f> source, std::vector<Eigen::Vector3f> target, float& scale);
  Eigen::Matrix4f SimilarityHorn(std::vector<Eigen::Vector3f> source, std::vector<Eigen::Vector3f> target, float& scale);
};

} // namespace fusion

#endif