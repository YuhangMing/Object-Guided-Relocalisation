#include "tracking/rgbd_odometry.h"
#include "tracking/icp_tracker.h"

namespace fusion
{

DenseOdometry::DenseOdometry(const fusion::IntrinsicMatrix base, int NUM_PYR)
    : tracker(new DenseTracking()),
      trackingLost(false),
      initialized(false)
{
  currDeviceMapPyramid = std::make_shared<DeviceImage>(base, NUM_PYR);
  std::shared_ptr<DeviceImage> refDeviceMapPyramid = std::make_shared<DeviceImage>(base, NUM_PYR);
  vModelDeviceMapPyramid.push_back(refDeviceMapPyramid);
  BuildIntrinsicPyramid(base, cam_params, NUM_PYR);
  this->base = base;
  this->NUM_PYR = NUM_PYR;
}

void DenseOdometry::trackFrame(std::shared_ptr<RgbdFrame> frame)
{
  // CURRENT, updated in every submap
  upload(frame);  // nmap & vmap calculated here

  if (!initialized)
  {
    // std::cout << "Odometry: Initializing... " << std::endl;
    vModelFrames.push_back(frame);
    copyDeviceImage(currDeviceMapPyramid, vModelDeviceMapPyramid[submapIdx]);
    initialized = true;
    return;
  }

  // std::cout << " In Tracking: current frame is: " << frame->id
  //           << ", reference frame is: " << vModelFrames[submapIdx]->id
  //           << std::endl;

  // std::cout << "Odometry: Set context... " << std::endl;
  context.use_initial_guess_ = true;
  context.initial_estimate_ = Sophus::SE3d();
  context.intrinsics_pyr_ = cam_params;
  context.max_iterations_ = {10, 5, 3, 3, 3};

  // std::cout << "Odometry: Compute transform... " << std::endl;
  if(manager->active_submaps[submapIdx]->bTrack){
    result = tracker->compute_transform(vModelDeviceMapPyramid[submapIdx], currDeviceMapPyramid, context);
    result.update = vModelFrames[submapIdx]->pose * result.update;
  } else {
    Sophus::SE3d Tmf = vModelFrames[trackIdx]->pose;
    Sophus::SE3d Twm = manager->active_submaps[trackIdx]->poseGlobal;
    Sophus::SE3d Twcinv = manager->active_submaps[submapIdx]->poseGlobal.inverse();
    // pose of input frame w.r.t. current sm
    result.update = Twcinv * Twm * Tmf;
    result.sucess = true;
  }

  // std::cout << "Odometry: Update LastFrame... " << std::endl;
  if (result.sucess)
  {
    // if(manager->active_submaps[submapIdx]->bTrack)
    //   frame->pose = vModelFrames[submapIdx]->pose * result.update;
    // else
    frame->pose = result.update;
    vModelFrames[submapIdx] = frame;
    copyDeviceImage(currDeviceMapPyramid, vModelDeviceMapPyramid[submapIdx]);
    trackingLost = false;
  }
  else
  {
    trackingLost = true;
  }
}

void DenseOdometry::trackDepthOnly(std::shared_ptr<RgbdFrame> frame, float& icp_error)
{
  upload(frame);

  context.use_initial_guess_ = true;
  context.initial_estimate_ = Sophus::SE3d();
  context.intrinsics_pyr_ = cam_params;
  context.max_iterations_ = {10, 5, 3, 3, 3};

  result = tracker->compute_transform_depth_only(vModelDeviceMapPyramid[submapIdx], currDeviceMapPyramid, context);
  result.update = vModelFrames[submapIdx]->pose * result.update;

  icp_error = result.icp_error;
  
  if (result.sucess)
  {
    frame->pose = result.update;
    vModelFrames[submapIdx] = frame;
    copyDeviceImage(currDeviceMapPyramid, vModelDeviceMapPyramid[submapIdx]);
    trackingLost = false;
  }
  else
  {
    frame->pose = result.update;
    trackingLost = true;
  }
}

void DenseOdometry::trackDepthAndCentroid(std::shared_ptr<RgbdFrame> frame, float& icp_error)
{
  upload(frame);

  // // TEST: display centroid 
  // for(size_t i=0; i<6; ++i)
  // {
  //   std::cout << "Label " << i << " map (" << vModelFrames[submapIdx]->cent_matrix.at<float>(i, 0)
  //             << ", " << vModelFrames[submapIdx]->cent_matrix.at<float>(i, 1)
  //             << ", " << vModelFrames[submapIdx]->cent_matrix.at<float>(i, 2) 
  //             << "); frame (" << frame->cent_matrix.at<float>(i, 0)
  //             << ", " << frame->cent_matrix.at<float>(i, 1)
  //             << ", " << frame->cent_matrix.at<float>(i, 2) << ")" << std::endl;
  // }

  context.use_initial_guess_ = true;
  context.initial_estimate_ = Sophus::SE3d();
  context.intrinsics_pyr_ = cam_params;
  context.max_iterations_ = {10, 5, 3, 3, 3};
  // level-0 image of original size, level-1 image of halved size, ...

  result = tracker->compute_transform_depth_centroids(vModelDeviceMapPyramid[submapIdx], currDeviceMapPyramid, context);
  result.update = vModelFrames[submapIdx]->pose * result.update;

  icp_error = result.icp_error;
  
  if (result.sucess)
  {
    frame->pose = result.update;
    vModelFrames[submapIdx] = frame;
    copyDeviceImage(currDeviceMapPyramid, vModelDeviceMapPyramid[submapIdx]);
    trackingLost = false;
  }
  else
  {
    // frame->pose = result.update; // invalid pose if the error is invalid
    frame->pose = Sophus::SE3d();
    trackingLost = true;
  }
}

std::shared_ptr<DeviceImage> DenseOdometry::get_current_image() const
{
  return currDeviceMapPyramid;
}

std::shared_ptr<DeviceImage> DenseOdometry::get_reference_image(int i) const
{
  return vModelDeviceMapPyramid[i];
}

void DenseOdometry::upload(std::shared_ptr<RgbdFrame> frame){
  currDeviceMapPyramid->upload(frame);
}

void DenseOdometry::upload_semantics(std::shared_ptr<RgbdFrame> frame, int i){
  if(!vModelDeviceMapPyramid.empty())
    vModelDeviceMapPyramid[i]->upload_semantics(frame);
  else
    currDeviceMapPyramid->upload_semantics(frame);
}

// Eigen::Matrix4f DenseOdometry::get_current_pose_matrix() const
  // {
  //   if (currDeviceMapPyramid && currDeviceMapPyramid->get_reference_frame())
  //   {
  //     return currDeviceMapPyramid->get_reference_frame()->pose.matrix().cast<float>();
  //   }
  //   else
  //     return Eigen::Matrix4f::Identity();
// }

void DenseOdometry::relocUpdate(std::shared_ptr<RgbdFrame> frame)
{
  std::shared_ptr<DeviceImage> currDeviceMapPyramid_copy = std::make_shared<DeviceImage>(base, NUM_PYR);
  currDeviceMapPyramid_copy->upload(frame);
  copyDeviceImage(currDeviceMapPyramid_copy, vModelDeviceMapPyramid[submapIdx]);
  // vModelFrames[submapIdx] = std::make_shared<RgbdFrame>();
  // frame->copyTo(vModelFrames[submapIdx]);
  if(submapIdx < vModelFrames.size()){
    vModelFrames[submapIdx] = frame;
  } else {
    vModelFrames.push_back(frame);
  }
}

void DenseOdometry::reset()
{
  vModelFrames.clear();
  initialized = false;
  trackingLost = false;
}

void DenseOdometry::SetManager(std::shared_ptr<SubMapManager> pManager){
  manager = pManager;
}

void DenseOdometry::setSubmapIdx(int idx){
  submapIdx = idx;
}

void DenseOdometry::setTrackIdx(int idx){
  trackIdx = idx;
}

// void DenseOdometry::SetDetector(semantic::MaskRCNN * pDetector){
//   detector = pDetector;
// }

} // namespace fusion