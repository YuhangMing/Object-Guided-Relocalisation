#ifndef SLAM_POSE_ESTIMATOR_H
#define SLAM_POSE_ESTIMATOR_H

#include "data_struct/map_point.h"

namespace fusion
{

class PoseEstimator
{
public:
    //! Compute relative transformation from two sets of points
    //! outliers: indicates which points are outliers
    //! At least 3 pairs of points need to be supplied
    //! and none of them can be co-linear
    static bool absolute_orientation(
        std::vector<Eigen::Vector3d> src,
        std::vector<Eigen::Vector3d> dst,
        std::vector<bool> outliers,
        Eigen::Matrix4d &estimate);

    //! Compute relative transformation from two sets of points
    //! all points are treated like inliers
    //! At least 3 pairs of points need to be supplied
    //! and none of them can be co-linear
    static bool absolute_orientation(
        std::vector<Eigen::Vector3d> src,
        std::vector<Eigen::Vector3d> dst,
        Eigen::Matrix4d &estimate);

    static int evaluate_inliers(
        const std::vector<Eigen::Vector3d> &src,
        const std::vector<Eigen::Vector3d> &dst,
        std::vector<double> weight,
        std::vector<bool> &outliers,
        const Eigen::Matrix4d &estimate,
        double &loss);

    static float compute_residual(
        const std::vector<Eigen::Vector3d> &src,
        const std::vector<Eigen::Vector3d> &dst,
        const Eigen::Matrix4d &pose_estimate);

    static bool RANSAC(
        const std::vector<Eigen::Vector3d> &src,
        const std::vector<Eigen::Vector3d> &dst,
        std::vector<double> &weight,
        std::vector<bool> &outliers,
        Eigen::Matrix4d &estimate,
        float &inlier_ratio,
        float &confidence);

private:
};

} // namespace fusion

#endif