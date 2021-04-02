#include "tracking/rgbd_odometry.h"
#include "tracking/icp_tracker.h"
#include "utils/settings.h"

namespace fusion
{

DenseOdometry::DenseOdometry()
    : tracker(new DenseTracking()),
      trackingLost(false),
      initialized(false)
{
  // Create K pyramid
  std::cout << "Inside DO: \n" << GlobalCfg.K << std::endl << std::endl;
  vK.clear();
  vKInv.clear();
  vK.push_back(GlobalCfg.K);
  vKInv.push_back(GlobalCfg.KInv);
  for (int i = 0; i < GlobalCfg.maxPyramidLevel - 1; ++i)
  {
    Eigen::Matrix3f tmpK = vK[i]*0.5;
    vK.push_back(tmpK);

    Eigen::Matrix3f tmpKInv = tmpK;
    tmpKInv(0,0) = 1.0/tmpKInv(0,0);
    tmpKInv(1,1) = 1.0/tmpKInv(1,1);
    vKInv.push_back(tmpKInv);
  }

  currDeviceMapPyramid = std::make_shared<DeviceImage>(vKInv);
  std::shared_ptr<DeviceImage> refDeviceMapPyramid = std::make_shared<DeviceImage>(vKInv);
  vModelDeviceMapPyramid.push_back(refDeviceMapPyramid);
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
  context.K_pyr_ = vK;
  context.max_iterations_ = {10, 5, 3, 3, 3};

  // std::cout << "Odometry: Compute transform... " << std::endl;
  // if(manager->active_submaps[submapIdx]->bTrack){
    result = tracker->compute_transform(vModelDeviceMapPyramid[submapIdx], currDeviceMapPyramid, context);
    result.update = vModelFrames[submapIdx]->pose * result.update;
  // } else {
  //   Sophus::SE3d Tmf = vModelFrames[trackIdx]->pose;
  //   Sophus::SE3d Twm = manager->active_submaps[trackIdx]->poseGlobal;
  //   Sophus::SE3d Twcinv = manager->active_submaps[submapIdx]->poseGlobal.inverse();
  //   // pose of input frame w.r.t. current sm
  //   result.update = Twcinv * Twm * Tmf;
  //   result.sucess = true;
  // }

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

void DenseOdometry::trackDepthAndCentroid(std::shared_ptr<RgbdFrame> frame, float& icp_error)
{
  upload(frame);
  // // TEST: display centroid 
  // for(size_t i=0; i<30; ++i)
  // {
  //   std::cout << "Label " << i%6+1 << " map (" << vModelFrames[submapIdx]->cent_matrix.at<float>(i, 0)
  //             << ", " << vModelFrames[submapIdx]->cent_matrix.at<float>(i, 1)
  //             << ", " << vModelFrames[submapIdx]->cent_matrix.at<float>(i, 2) 
  //             << "); frame (" << frame->cent_matrix.at<float>(i, 0)
  //             << ", " << frame->cent_matrix.at<float>(i, 1)
  //             << ", " << frame->cent_matrix.at<float>(i, 2) << ")" << std::endl;
  // }

  context.use_initial_guess_ = true;
  context.initial_estimate_ = Sophus::SE3d();
  context.K_pyr_ = vK;
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

/* Semantic & Reloc disabled for now.
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
*/


void DenseOdometry::reset()
{
  vModelFrames.clear();
  initialized = false;
  trackingLost = false;
}

void DenseOdometry::relocUpdate(std::shared_ptr<RgbdFrame> frame)
{
  // ~0.003s
  // update model
  std::shared_ptr<DeviceImage> currDeviceMapPyramid_copy = std::make_shared<DeviceImage>(vKInv);
  currDeviceMapPyramid_copy->upload(frame);
  copyDeviceImage(currDeviceMapPyramid_copy, vModelDeviceMapPyramid[submapIdx]);
  if(submapIdx < vModelFrames.size()){
    vModelFrames[submapIdx] = frame;
  } else {
    vModelFrames.push_back(frame);
  }
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


std::vector<Sophus::SE3d> DenseOdometry::get_keyframe_poses() const
{}

std::vector<Sophus::SE3d> DenseOdometry::get_camera_trajectory() const
{}

// Eigen::Matrix4f DenseOdometry::get_current_pose_matrix() const
//   {
//     if (currDeviceMapPyramid && currDeviceMapPyramid->get_reference_frame())
//     {
//       return currDeviceMapPyramid->get_reference_frame()->pose.matrix().cast<float>();
//     }
//     else
//       return Eigen::Matrix4f::Identity();
// }

std::shared_ptr<DeviceImage> DenseOdometry::get_current_image() const
{
  return currDeviceMapPyramid;
}

std::shared_ptr<DeviceImage> DenseOdometry::get_reference_image(int i) const
{
  return vModelDeviceMapPyramid[i];
}


void DenseOdometry::SetManager(std::shared_ptr<SubmapManager> pManager){
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

/* Attepmt to not use device image in tracking. Result in noisy reconstruction.
void DenseOdometry::trackFrame(std::shared_ptr<RgbdFrame> frame)
{
  // // CURRENT, updated in every submap
  // upload(frame);  // nmap & vmap calculated here

  if (!initialized)
  {
    std::cout << "Odometry: Initialising... " << std::endl;
    vModelFrames.push_back(frame);
    
    std::cout << "-- Constructing intensity pyramid...";
    // tracker->set_source_image(frame->image, vKInv.size());
    tracker->set_reference_image(frame->image, vKInv.size());
    std::cout << "  Done.\n" << "-- Constructing vmap and nmap pyramid..." ;
    // tracker->set_source_depth(frame->depth, vKInv);
    tracker->set_reference_depth(frame->depth, vKInv);
    std::cout << "  Done." << std::endl;

    initialized = true;
    return;
  }

  // std::cout << " In Tracking: current frame is: " << frame->id
  //           << ", reference frame is: " << vModelFrames[submapIdx]->id
  //           << std::endl;
  std::cout << "Odometry: Tracking... " << std::endl;

  // std::cout << "Odometry: Set context... " << std::endl;
  context.use_initial_guess_ = true;
  context.initial_estimate_ = Sophus::SE3d();
  context.K_pyr_ = vK;
  context.max_iterations_ = {10, 5, 3, 3, 3};

  // std::cout << "Odometry: Compute transform... " << std::endl;

  
  // intensity_src_pyr, intensity_ref_pyr, intensity_dx_pyr, intensity_dy_pyr.
  std::cout << "-- Constructing intensity pyramid...";
  tracker->set_source_image(frame->image, vKInv.size());
  // vmap_src_pyr, vmap_ref_pyr, nmap_src_pyr, nmap_ref_pyr;
  std::cout << "  Done.\n" << "-- Constructing vmap and nmap pyramid..." ;
  tracker->set_source_depth(frame->depth, vKInv);
  std::cout << "  Done." << std::endl;
  
  // Tracking
  result = tracker->compute_transform(context);
  // std::cout << "Reference frame pose: \n" << vModelFrames[submapIdx]->pose.matrix() << std::endl;
  // std::cout << "Incremental transformation: \n" << result.update.matrix() << std::endl;
  result.update = vModelFrames[submapIdx]->pose * result.update;

  // if(manager->active_submaps[submapIdx]->bTrack)
  // {
  //   result = tracker->compute_transform(context);
  //   result.update = vModelFrames[submapIdx]->pose * result.update;
  // } 
  // else 
  // {
  //   Sophus::SE3d Tmf = vModelFrames[trackIdx]->pose;
  //   Sophus::SE3d Twm = manager->active_submaps[trackIdx]->poseGlobal;
  //   Sophus::SE3d Twcinv = manager->active_submaps[submapIdx]->poseGlobal.inverse();
  //   // pose of input frame w.r.t. current sm
  //   result.update = Twcinv * Twm * Tmf;
  //   result.sucess = true;
  // }

  // std::cout << "Odometry: Update LastFrame... " << std::endl;
  if (result.sucess)
  {
    // if(manager->active_submaps[submapIdx]->bTrack)
    //   frame->pose = vModelFrames[submapIdx]->pose * result.update;
    // else
    frame->pose = result.update;
    vModelFrames[submapIdx] = frame;
    trackingLost = false;

    std::cout << frame->pose.matrix() << std::endl;
  }
  else
  {
    trackingLost = true;
  }
}

cv::cuda::GpuMat DenseOdometry::get_current_color(){
  return tracker->get_image_src();
}

cv::cuda::GpuMat DenseOdometry::get_current_depth(){
  return tracker->get_depth_src();
}

cv::cuda::GpuMat DenseOdometry::get_current_vmap(const int &level){
  return tracker->get_vmap_src(level);
}

cv::cuda::GpuMat DenseOdometry::get_current_nmap(const int &level){
  return tracker->get_nmap_src(level);
}

void DenseOdometry::update_reference_model(cv::cuda::GpuMat vmap){
  std::cout << "# Source vmap address (before updating): " << &vmap << std::endl;
  // both vmap and nmap in tracker are updated
  std::cout << "Updating reference vmap and nmap pyramid." << std::endl;
  tracker->update_reference_vnmap(vmap);
  std::cout << "Updating reference intensity pyramid, image, and depth pyramid (default false)." << std::endl;
  tracker->update_ref_with_src(true, true, true);
  std::cout << "Done" << std::endl;
}
*/

} // namespace fusion