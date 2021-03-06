#ifndef SLAM_LIB_DENSE_ODOMETRY_H
#define SLAM_LIB_DENSE_ODOMETRY_H

#include "data_struct/rgbd_frame.h"
#include "tracking/device_image.h"
#include "tracking/icp_tracker.h"
#include "tracking/cuda_imgproc.h"
#include "voxel_hashing/voxel_hashing.h"
#include "map_manager.h"
#include "detection/detector.h"
#include <memory>

namespace fusion
{

class SubMapManager;
class DenseMapping;
// class MaskRCNN;

class DenseOdometry
{
public:
  DenseOdometry(const fusion::IntrinsicMatrix base, int NUM_PYR);
  DenseOdometry(const DenseOdometry &) = delete;
  DenseOdometry &operator=(const DenseOdometry &) = delete;

  bool trackingLost;
  void trackFrame(std::shared_ptr<RgbdFrame> frame);
  void trackDepthOnly(std::shared_ptr<RgbdFrame> frame, float& icp_error);
  void trackDepthAndCentroid(std::shared_ptr<RgbdFrame> frame, float& icp_error);
  void reset();
  void relocUpdate(std::shared_ptr<RgbdFrame> frame);

  void upload(std::shared_ptr<RgbdFrame> frame);
  void upload_semantics(std::shared_ptr<RgbdFrame> frame, int i);

  std::vector<Sophus::SE3d> get_keyframe_poses() const;
  std::vector<Sophus::SE3d> get_camera_trajectory() const;

  // Eigen::Matrix4f get_current_pose_matrix() const;
  std::shared_ptr<DeviceImage> get_current_image() const;
  std::shared_ptr<DeviceImage> get_reference_image(int i) const;

  // submap related
  void SetManager(std::shared_ptr<SubMapManager> pManager);
  void setSubmapIdx(int idx);
  void setTrackIdx(int idx);

  std::vector< std::shared_ptr<RgbdFrame> > vModelFrames;    // or condsider move this to system, only keep mappyramid here
  std::vector< std::shared_ptr<DeviceImage> > vModelDeviceMapPyramid;

  // // semantic related
  // void SetDetector(semantic::MaskRCNN * pDetector);

private:
  // std::shared_ptr<RgbdFrame> lastTracedFrame;         // consider change this to vector
  std::shared_ptr<DeviceImage> currDeviceMapPyramid;
  // std::shared_ptr<DeviceImage> refDeviceMapPyramid;   // consider change this to vector
  std::vector<fusion::IntrinsicMatrix> cam_params;
  std::unique_ptr<DenseTracking> tracker;
  bool initialized;

  TrackingResult result;
  TrackingContext context;

  // submap related
  std::shared_ptr<SubMapManager> manager;
  int submapIdx;    // index of current processing submap (among active submaps)
  int trackIdx;     // index of last rendered submap (among active submaps)

  // // semantic related
  // semantic::MaskRCNN * detector;

  // for reloc
  IntrinsicMatrix base;
  int NUM_PYR;
};

} // namespace fusion

#endif