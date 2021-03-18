#ifndef FUSION_DESCRIPTOR_MATCHER_H
#define FUSION_DESCRIPTOR_MATCHER_H

#include <thread>
#include <opencv2/xfeatures2d.hpp>
#include "data_struct/map_point.h"
#include "data_struct/rgbd_frame.h"
#include "data_struct/intrinsic_matrix.h"

namespace fusion
{

class DescriptorMatcher
{
public:
    DescriptorMatcher();

    void match_hamming_knn(
        const cv::Mat trainDesc,
        const cv::Mat queryDesc,
        std::vector<std::vector<cv::DMatch>> &matches,
        const int k = 2);

    std::thread match_hamming_knn_async(
        const cv::Mat trainDesc,
        const cv::Mat queryDesc,
        std::vector<std::vector<cv::DMatch>> &matches,
        const int k = 2);

    void filter_matches_pair_constraint(
        const std::vector<std::shared_ptr<Point3d>> &src_pts,
        const std::vector<std::shared_ptr<Point3d>> &dst_pts,
        const std::vector<std::vector<cv::DMatch>> &knnMatches,
        std::vector<std::vector<cv::DMatch>> &candidates);

    void filter_matches_ratio_test(
        const std::vector<std::vector<cv::DMatch>> &knnMatches,
        std::vector<cv::DMatch> &candidates);

    void match_pose_constraint(
        RgbdFramePtr source,
        RgbdFramePtr reference,
        const fusion::IntrinsicMatrix &cam_params,
        const Sophus::SE3f &pose);

private:
    cv::Ptr<cv::DescriptorMatcher> hammingMatcher;
    cv::Ptr<cv::DescriptorMatcher> l2Matcher;
};

} // namespace fusion

#endif