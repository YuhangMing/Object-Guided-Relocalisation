#include "tracking/build_pyramid.h"
#include "tracking/pose_estimator.h"
#include "tracking/m_estimator.h"
#include "utils/revertable.h"
#include "data_struct/rgbd_frame.h"
#include "tracking/icp_tracker.h"
#include "tracking/device_image.h"
#include "utils/safe_call.h"

namespace fusion
{

DenseTracking::DenseTracking()
{
  safe_call(cudaGetLastError());
  SUM_SE3.create(96, 29, CV_32FC1);
  safe_call(cudaGetLastError());
  OUT_SE3.create(1, 29, CV_32FC1);
}

DenseTracking::DenseTracking(const IntrinsicMatrix K, const int NUM_PYR) : DenseTracking()
{
  BuildIntrinsicPyramid(K, cam_params, NUM_PYR);
}

TrackingResult DenseTracking::compute_transform(const RgbdImagePtr reference, const RgbdImagePtr current, const TrackingContext &c)
{
  Revertable<Sophus::SE3d> estimate = Revertable<Sophus::SE3d>(Sophus::SE3d());

  if (c.use_initial_guess_)
    estimate = Revertable<Sophus::SE3d>(c.initial_estimate_);

  bool invalid_error = false;
  Sophus::SE3d init_estimate = estimate.get();

  for (int level = c.max_iterations_.size() - 1; level >= 0; --level)
  {
    cv::cuda::GpuMat curr_vmap = current->get_vmap(level);
    cv::cuda::GpuMat last_vmap = reference->get_vmap(level);
    cv::cuda::GpuMat curr_nmap = current->get_nmap(level);
    cv::cuda::GpuMat last_nmap = reference->get_nmap(level);
    cv::cuda::GpuMat curr_intensity = current->get_intensity(level);
    cv::cuda::GpuMat last_intensity = reference->get_intensity(level);
    cv::cuda::GpuMat intensity_dx = current->get_intensity_dx(level);
    cv::cuda::GpuMat intensity_dy = current->get_intensity_dy(level);
    IntrinsicMatrix K = c.intrinsics_pyr_[level];
    float icp_error = std::numeric_limits<float>::max();
    float rgb_error = std::numeric_limits<float>::max();
    float total_error = std::numeric_limits<float>::max();
    int icp_count = 0, rgb_count = 0;
    float stddev_estimated = 0;

    for (int iter = 0; iter < c.max_iterations_[level]; ++iter)
    {
      auto last_estimate = estimate.get();
      auto last_icp_error = icp_error;
      auto last_rgb_error = rgb_error;

      icp_reduce(
          curr_vmap,
          curr_nmap,
          last_vmap,
          last_nmap,
          SUM_SE3,
          OUT_SE3,
          last_estimate,
          K,
          icp_hessian.data(),
          icp_residual.data(),
          residual_icp_.data());

      float stdev_estimated;

      rgb_step(
          curr_intensity,
          last_intensity,
          last_vmap,
          curr_vmap,
          intensity_dx,
          intensity_dy,
          SUM_SE3,
          OUT_SE3,
          stddev_estimated,
          last_estimate,
          K,
          rgb_hessian.data(),
          rgb_residual.data(),
          residual_rgb_.data());

      stddev_estimated = sqrt(residual_rgb_[0] / (residual_rgb_[1] - 6));

      auto A = 1e6 * icp_hessian + rgb_hessian;
      auto b = 1e6 * icp_residual + rgb_residual;

      update = A.cast<double>().ldlt().solve(b.cast<double>());

      estimate = Sophus::SE3d::exp(update) * last_estimate; 
      icp_error = sqrt(residual_icp_(0)) / residual_icp_(1);

      if (std::isnan(icp_error))
      {
        invalid_error = true;
        break;
      }

      if (icp_error > last_icp_error)
      {
        if (icp_count >= 2)
        {
          estimate.revert();
          break;
        }

        icp_count++;
        icp_error = last_icp_error;
      }
      else
      {
        icp_count = 0;
      }

      rgb_error = sqrt(residual_rgb_(0)) / residual_rgb_(1);

      if (std::isnan(rgb_error))
      {
        invalid_error = true;
        break;
      }

      if (rgb_error > last_rgb_error)
      {
        if (rgb_count >= 2)
        {
          estimate.revert();
          break;
        }

        rgb_count++;
        rgb_error = last_rgb_error;
      }
      else
      {
        rgb_count = 0;
      }
    }
  }

  TrackingResult result;

  if (invalid_error || (estimate.get().inverse() * init_estimate).log().norm() > 0.1)
  {
    result.sucess = false;
  }
  else
  {
    result.sucess = true;
    result.update = estimate.get().inverse();
  }

  return result;
}

TrackingResult DenseTracking::compute_transform_depth_only(const RgbdImagePtr reference, const RgbdImagePtr current, const TrackingContext &c)
{
  Revertable<Sophus::SE3d> estimate = Revertable<Sophus::SE3d>(Sophus::SE3d());

  if (c.use_initial_guess_)
    estimate = Revertable<Sophus::SE3d>(c.initial_estimate_);

  bool invalid_error = false;
  Sophus::SE3d init_estimate = estimate.get();
  float final_icp_error = 0;

  for (int level = c.max_iterations_.size() - 1; level >= 0; --level)
  {
    cv::cuda::GpuMat curr_vmap = current->get_vmap(level);
    cv::cuda::GpuMat last_vmap = reference->get_vmap(level);
    cv::cuda::GpuMat curr_nmap = current->get_nmap(level);
    cv::cuda::GpuMat last_nmap = reference->get_nmap(level);
    IntrinsicMatrix K = c.intrinsics_pyr_[level];
    float icp_error = std::numeric_limits<float>::max();
    int icp_count = 0;
    float stddev_estimated = 0;

    for (int iter = 0; iter < c.max_iterations_[level]; ++iter)
    {
      auto last_estimate = estimate.get();
      auto last_icp_error = icp_error;

      icp_reduce(
          curr_vmap,
          curr_nmap,
          last_vmap,
          last_nmap,
          SUM_SE3,
          OUT_SE3,
          last_estimate,
          K,
          icp_hessian.data(),
          icp_residual.data(),
          residual_icp_.data());

      update = icp_hessian.cast<double>().ldlt().solve(icp_residual.cast<double>());
      estimate = Sophus::SE3d::exp(update) * last_estimate;

      icp_error = sqrt(residual_icp_(0)) / residual_icp_(1);

      // std::cout << "-- icp error: " << residual_icp_(0) << " - "
      //           << residual_icp_(1) << " - " << icp_error
      //           << std::endl;

      if (std::isnan(icp_error))
      {
        std::cout << "!! Invalid icp error at level-" << level 
                  << ", iter-" << iter << std::endl;
        invalid_error = true;
        break;
      }

      if (icp_error > last_icp_error)
      {
        if (icp_count >= 2)
        {
          estimate.revert();
          break;
        }

        icp_count++;
        icp_error = last_icp_error;
        final_icp_error = icp_error;
      }
      else
      {
        icp_count = 0;
      }

    }
  }

  TrackingResult result;

  // if (invalid_error || (estimate.get().inverse() * init_estimate).log().norm() > 0.2)
  if (invalid_error)
  
  {
    result.sucess = false;
    result.icp_error =  std::numeric_limits<float>::max();
  }
  else
  {
    result.sucess = true;
    result.icp_error = final_icp_error;
    result.update = estimate.get().inverse();
  }

  return result;
}

TrackingResult DenseTracking::compute_transform_depth_centroids(const RgbdImagePtr reference, const RgbdImagePtr current, const TrackingContext &c)
{
  Revertable<Sophus::SE3d> estimate = Revertable<Sophus::SE3d>(Sophus::SE3d());

  if (c.use_initial_guess_)
    estimate = Revertable<Sophus::SE3d>(c.initial_estimate_);

  bool invalid_error = false;
  Sophus::SE3d init_estimate = estimate.get();
  float w_icp = 1.,
        w_cent = 1.;
  float final_icp_error = 0.,
        final_cent_error = 0.;

  cv::Mat curr_cent = current->get_centroids();
  cv::Mat last_cent = reference->get_centroids();
  for (int level = c.max_iterations_.size() - 1; level >= 0; --level)
  {
    cv::cuda::GpuMat curr_vmap = current->get_vmap(level);
    cv::cuda::GpuMat last_vmap = reference->get_vmap(level);
    cv::cuda::GpuMat curr_nmap = current->get_nmap(level);
    cv::cuda::GpuMat last_nmap = reference->get_nmap(level);
    IntrinsicMatrix K = c.intrinsics_pyr_[level];
    float icp_error = std::numeric_limits<float>::max();
    float cent_error = std::numeric_limits<float>::max();
    float total_error = std::numeric_limits<float>::max();
    int icp_count = 0, cent_count = 0;
    float stddev_estimated = 0;

    // std::cout << "Recude on level-" << level << std::endl;
    for (int iter = 0; iter < c.max_iterations_[level]; ++iter)
    {
      auto last_estimate = estimate.get();
      auto last_icp_error = icp_error;
      auto last_cent_error = cent_error;

      icp_reduce(
          curr_vmap,
          curr_nmap,
          last_vmap,
          last_nmap,
          SUM_SE3,
          OUT_SE3,
          last_estimate,
          K,
          icp_hessian.data(),
          icp_residual.data(),
          residual_icp_.data());
      // std::cout << "icp: \n" << icp_hessian << std::endl << icp_residual << std::endl;

      cent_reduce(
          curr_cent,
          last_cent,
          last_estimate,
          cent_hessian.data(),
          cent_residual.data(),
          residual_cent_.data());
      // std::cout << "cent: \n" << cent_hessian << std::endl << cent_residual << std::endl; 

      auto A = w_icp*icp_hessian + w_cent*cent_hessian;
      auto b = w_icp*icp_residual + w_cent*cent_residual;

      update = A.cast<double>().ldlt().solve(b.cast<double>());
      estimate = Sophus::SE3d::exp(update) * last_estimate;

      icp_error = sqrt(residual_icp_(0)) / residual_icp_(1);
      cent_error = sqrt(residual_cent_(0)) / residual_cent_(1);
      std::cout << "--error: icp-" << icp_error << ", cent-" << cent_error << std::endl;

      if (std::isnan(icp_error) || std::isnan(cent_error))
      {
        invalid_error = true;
        break;
      }

      if (icp_error > last_icp_error)
      {
        if (icp_count >= 2)
        {
          estimate.revert();
          // std::cout << "!! Early stop triggered at icp error." << std::endl;
          break;
        }

        icp_count++;
        icp_error = last_icp_error;
        final_icp_error = icp_error;
      }
      else
      {
        icp_count = 0;
      }

      if (cent_error >= last_cent_error)
      {
        if (cent_count >= 2)
        {
          estimate.revert();
          // std::cout << "!! Early stop triggered at cent error." << std::endl;
          break;
        }

        cent_count++;
        cent_error = last_cent_error;
        final_cent_error = cent_error;
      }
      else
      {
        cent_count = 0;
      }
    }
  }

  TrackingResult result;

  if (invalid_error)
  {
    result.sucess = false;
    result.icp_error = std::numeric_limits<float>::max();
  }
  else
  {
    result.sucess = true;
    result.icp_error = w_icp*final_icp_error + w_cent*final_cent_error;
    result.update = estimate.get().inverse();
  }

  return result;
}
void DenseTracking::cent_reduce(const cv::Mat &curr_cent, const cv::Mat &last_cent,
                                const Sophus::SE3d &pose,
                                float *jtj, float *jtr, float *residual)
{
  /* compute J^TJ and J^Tr */

  // loop through all the centoids to compute values
  // std::cout << "looping through all the centroids.. " << std::endl;
  float sum[29] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0};
  for(size_t i=0; i<30; ++i)
  {
    Eigen::Matrix3f R = pose.matrix().cast<float>().topLeftCorner(3,3);
    Eigen::Vector3f t = pose.matrix().cast<float>().topRightCorner(3,1);
    Eigen::Vector3f clast, ccurr;
    clast << last_cent.at<float>(i, 0), last_cent.at<float>(i, 1), last_cent.at<float>(i, 2);
    // std::cout << clast << std::endl;
    if(clast(0)==0 && clast(1)==0 && clast(2)==0)
      continue;
    else
      clast = R * clast + t;
    // std::cout << clast << std::endl;

    ccurr << curr_cent.at<float>(i, 0), curr_cent.at<float>(i, 1), curr_cent.at<float>(i, 2);
    // std::cout << ccurr << std::endl;
    if(ccurr(0)==0 && ccurr(1)==0 && ccurr(2)==0)
      continue;
    
    Eigen::Vector3f ccross = clast.cross(ccurr);
    Eigen::Vector3f cdiff = ccurr - clast;
    float val[29] = { 1, 0, 0, 0, clast(2), -1*clast(1), cdiff(0),
                      1, 0, -1*clast(2), 0, clast(0), cdiff(1),
                      1, clast(1), -1*clast(0), 0, cdiff(2),
                      clast(2)*clast(2)+clast(1)*clast(1), -1*clast(0)*clast(1), -1*clast(0)*clast(2), ccross(0),
                      clast(2)*clast(2)+clast(0)*clast(0), -1*clast(1)*clast(2), ccross(1),
                      clast(0)*clast(0)+clast(1)*clast(1), ccross(2),
                      cdiff(0)*cdiff(0)+cdiff(1)*cdiff(1)+cdiff(2)*cdiff(2),
                      1};

    for(size_t j=0; j<29; ++j)
      sum[j] += val[j];
  }
  
  // create JtJ and Jtr matrix
  // std::cout << "creating JtJ and Jtr matrices.. " << std::endl;
  int shift=0;
  for(size_t i=0; i<6; ++i)
  {
    for(size_t j=i; j<7; ++j)
    {
      float value = sum[shift++];
      if(j == 6)
        jtr[i] = value;
      else
        jtj[j*6+i] = jtj[i*6+j] = value;
    }
  }

  // retrieve cost and count
  // std::cout << "retrieving cost and count.. " << std::endl;
  residual[0] = sum[27];
  residual[1] = sum[28];
}


void DenseTracking::swap_intensity_pyr()
{
  for (int i = 0; i < intensity_src_pyr.size(); ++i)
    intensity_src_pyr[i].swap(intensity_ref_pyr[i]);
}

void DenseTracking::set_source_vmap(cv::cuda::GpuMat vmap)
{
  fusion::build_vmap_pyr(vmap, vmap_src_pyr, cam_params.size());
  fusion::build_nmap_pyr(vmap_src_pyr, nmap_src_pyr);
}

void DenseTracking::set_source_image(cv::cuda::GpuMat image)
{
  cv::cuda::GpuMat intensity;
  cv::cuda::cvtColor(image, intensity, cv::COLOR_RGB2GRAY);
  intensity.convertTo(intensity, CV_32FC1);
  set_source_intensity(intensity);
}

void DenseTracking::set_source_depth(cv::cuda::GpuMat depth_float)
{
  fusion::build_depth_pyr(depth_float, depth_src_pyr, cam_params.size());
}

void DenseTracking::set_source_intensity(cv::cuda::GpuMat intensity)
{
  fusion::build_intensity_pyr(intensity, intensity_src_pyr, cam_params.size());
  fusion::build_intensity_dxdy_pyr(intensity_src_pyr, intensity_dx_pyr, intensity_dy_pyr);
}

void DenseTracking::set_reference_image(cv::cuda::GpuMat image)
{
  cv::cuda::GpuMat intensity;
  cv::cuda::cvtColor(image, intensity, cv::COLOR_RGB2GRAY);
  intensity.convertTo(intensity, CV_32FC1);
  set_reference_intensity(intensity);
}

void DenseTracking::set_reference_intensity(cv::cuda::GpuMat intensity)
{
  fusion::build_intensity_pyr(intensity, intensity_ref_pyr, cam_params.size());
}

void DenseTracking::set_reference_vmap(cv::cuda::GpuMat vmap)
{
  fusion::build_vmap_pyr(vmap, vmap_ref_pyr, cam_params.size());
  fusion::build_nmap_pyr(vmap_ref_pyr, nmap_ref_pyr);
}

TrackingResult DenseTracking::compute_transform(const TrackingContext &context)
{
  Revertable<Sophus::SE3d> estimate = Revertable<Sophus::SE3d>(Sophus::SE3d());

  if (context.use_initial_guess_)
    estimate = Revertable<Sophus::SE3d>(context.initial_estimate_);

  for (int level = context.max_iterations_.size() - 1; level >= 0; --level)
  {
    cv::cuda::GpuMat curr_vmap = vmap_src_pyr[level];
    cv::cuda::GpuMat last_vmap = vmap_ref_pyr[level];
    cv::cuda::GpuMat curr_nmap = nmap_src_pyr[level];
    cv::cuda::GpuMat last_nmap = nmap_ref_pyr[level];
    cv::cuda::GpuMat curr_intensity = intensity_src_pyr[level];
    cv::cuda::GpuMat last_intensity = intensity_ref_pyr[level];
    cv::cuda::GpuMat intensity_dx = intensity_dx_pyr[level];
    cv::cuda::GpuMat intensity_dy = intensity_dy_pyr[level];
    IntrinsicMatrix K = cam_params[level];
    float icp_error = std::numeric_limits<float>::max();
    float rgb_error = std::numeric_limits<float>::max();
    float total_error = std::numeric_limits<float>::max();
    int icp_count = 0, rgb_count = 0;
    float stddev_estimated = 0;

    for (int iter = 0; iter < context.max_iterations_[level]; ++iter)
    {
      auto last_estimate = estimate.get();
      last_icp_error = icp_error;
      last_rgb_error = rgb_error;

      icp_reduce(
          curr_vmap,
          curr_nmap,
          last_vmap,
          last_nmap,
          SUM_SE3,
          OUT_SE3,
          last_estimate,
          K,
          icp_hessian.data(),
          icp_residual.data(),
          residual_icp_.data());

      float stdev_estimated;

      rgb_step(
          curr_intensity,
          last_intensity,
          last_vmap,
          curr_vmap,
          intensity_dx,
          intensity_dy,
          SUM_SE3,
          OUT_SE3,
          stddev_estimated,
          last_estimate,
          K,
          rgb_hessian.data(),
          rgb_residual.data(),
          residual_rgb_.data());

      stddev_estimated = sqrt(residual_rgb_[0] / (residual_rgb_[1] - 6));

      auto A = 1e6 * icp_hessian + rgb_hessian;
      auto b = 1e6 * icp_residual + rgb_residual;
      // auto A = icp_hessian;
      // auto b = icp_residual;

      update = A.cast<double>().ldlt().solve(b.cast<double>());
      estimate = Sophus::SE3d::exp(update) * last_estimate;

      icp_error = sqrt(residual_icp_(0)) / residual_icp_(1);

      if (icp_error > last_icp_error)
      {
        if (icp_count >= 2)
        {
          estimate.revert();
          break;
        }

        icp_count++;
        icp_error = last_icp_error;
      }
      else
      {
        icp_count = 0;
      }

      rgb_error = sqrt(residual_rgb_(0)) / residual_rgb_(1);

      if (rgb_error > last_rgb_error)
      {
        if (rgb_count >= 2)
        {
          estimate.revert();
          break;
        }

        rgb_count++;
        rgb_error = last_rgb_error;
      }
      else
      {
        rgb_count = 0;
      }
    }
  }

  // if (estimate.get().log().transpose().norm() > 0.1)
  //   std::cout << estimate.get().log().transpose().norm() << std::endl;

  TrackingResult result;
  result.sucess = true;
  result.icp_error = last_icp_error;
  result.update = estimate.get().inverse();
  return result;
}

cv::cuda::GpuMat DenseTracking::get_vmap_src(const int &level)
{
  return vmap_src_pyr[level];
}

cv::cuda::GpuMat DenseTracking::get_nmap_src(const int &level)
{
  return nmap_src_pyr[level];
}

cv::cuda::GpuMat DenseTracking::get_depth_src(const int &level)
{
  return depth_src_pyr[level];
}

cv::cuda::GpuMat DenseTracking::get_intensity_src(const int &level)
{
  return intensity_src_pyr[level];
}

cv::cuda::GpuMat DenseTracking::get_intensity_dx(const int &level)
{
  return intensity_dx_pyr[level];
}

cv::cuda::GpuMat DenseTracking::get_intensity_dy(const int &level)
{
  return intensity_dy_pyr[level];
}

cv::cuda::GpuMat DenseTracking::get_vmap_ref(const int &level)
{
  return vmap_ref_pyr[level];
}

cv::cuda::GpuMat DenseTracking::get_nmap_ref(const int &level)
{
  return nmap_ref_pyr[level];
}

cv::cuda::GpuMat DenseTracking::get_intensity_ref(const int &level)
{
  return intensity_ref_pyr[level];
}

} // namespace fusion