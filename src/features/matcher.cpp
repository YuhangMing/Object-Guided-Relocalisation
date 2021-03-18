#include "features/matcher.h"

namespace fusion
{

DescriptorMatcher::DescriptorMatcher()
{
    l2Matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_SL2);
    hammingMatcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
}

void DescriptorMatcher::match_hamming_knn(const cv::Mat trainDesc, const cv::Mat queryDesc, std::vector<std::vector<cv::DMatch>> &matches, const int k)
{
    hammingMatcher->knnMatch(queryDesc, trainDesc, matches, k);
}

std::thread DescriptorMatcher::match_hamming_knn_async(const cv::Mat trainDesc, const cv::Mat queryDesc, std::vector<std::vector<cv::DMatch>> &matches, const int k)
{
    return std::thread(&DescriptorMatcher::match_hamming_knn, this, trainDesc, queryDesc, std::ref(matches), k);
}

void DescriptorMatcher::filter_matches_pair_constraint(
    const std::vector<std::shared_ptr<Point3d>> &src_pts,
    const std::vector<std::shared_ptr<Point3d>> &dst_pts,
    const std::vector<std::vector<cv::DMatch>> &knnMatches,
    std::vector<std::vector<cv::DMatch>> &candidates)
{
    std::vector<cv::DMatch> rawMatch;
    candidates.clear();
    for (const auto &match : knnMatches)
    {
        if (match[0].distance / match[1].distance < 0.6)
        {
            rawMatch.push_back(std::move(match[0]));
        }
        else
        {
            rawMatch.push_back(std::move(match[0]));
            rawMatch.push_back(std::move(match[1]));
        }
    }

    const int NUM_RAW_MATCHES = rawMatch.size();
    cv::Mat adjecencyMat = cv::Mat::zeros(NUM_RAW_MATCHES, NUM_RAW_MATCHES, CV_32FC1);

    for (int y = 0; y < adjecencyMat.rows; ++y)
    {
        float *row = adjecencyMat.ptr<float>(y);
        const auto &match_y = rawMatch[y];
        const auto &match_y_src = src_pts[match_y.queryIdx];
        const auto &match_y_dst = dst_pts[match_y.trainIdx];

        for (int x = 0; x < adjecencyMat.cols; ++x)
        {
            const auto &match_x = rawMatch[x];
            const auto &match_x_src = src_pts[match_x.queryIdx];
            const auto &match_x_dst = dst_pts[match_x.trainIdx];

            if (match_x.trainIdx == match_y.trainIdx || match_x.queryIdx == match_y.queryIdx)
                continue;

            if (x == y)
            {
                row[x] = std::exp(-cv::norm(match_x_src->descriptors, match_x_dst->descriptors, cv::NORM_HAMMING));
            }
            else if (y < x)
            {

                const float src_dist = (match_x_src->pos - match_y_src->pos).norm();
                const float src_angle = std::acos(match_x_src->vec_normal.dot(match_y_src->vec_normal));

                const float dst_dist = (match_x_dst->pos - match_y_dst->pos).norm();
                const float dst_angle = std::acos(match_x_dst->vec_normal.dot(match_y_dst->vec_normal));

                float score = std::exp(-(std::fabs(src_dist - dst_dist) + std::fabs(src_angle - dst_angle)));
                if (std::isnan(score))
                    score = 0;

                row[x] = score;
            }
            else
            {
                row[x] = adjecencyMat.ptr<float>(x)[y];
            }
        }
    }

    cv::Mat reducedAM;
    cv::reduce(adjecencyMat, reducedAM, 0, cv::ReduceTypes::REDUCE_SUM);
    cv::Mat idxMat;
    cv::sortIdx(reducedAM, idxMat, cv::SortFlags::SORT_DESCENDING);

    std::vector<int> idxList;
    for (int y = 0; y < 1; ++y)
    {
        std::vector<cv::DMatch> selectedMatches;
        int head_idx = -1;
        size_t num_selected = 0;
        for (int x = y; x < idxMat.cols; ++x)
        {
            const auto &idx = idxMat.ptr<int>(0)[x];

            if (head_idx < 0)
            {
                head_idx = idx;
                selectedMatches.push_back(rawMatch[idx]);
                num_selected += 1;
            }
            else
            {
                const float &score = adjecencyMat.ptr<float>(head_idx)[idx];
                if (score > 0.1f)
                {
                    selectedMatches.push_back(rawMatch[idx]);
                    num_selected += 1;
                }
            }

            if (num_selected >= 200)
            {
                break;
            }
        }

        candidates.push_back(selectedMatches);
    }
}

void DescriptorMatcher::filter_matches_ratio_test(
    const std::vector<std::vector<cv::DMatch>> &knnMatches,
    std::vector<cv::DMatch> &candidates)
{
    candidates.clear();
    for (const auto &match : knnMatches)
    {
        if (match[0].distance / match[1].distance <= 0.75f)
        {
            candidates.push_back(match[0]);
        }
    }
}

void DescriptorMatcher::match_pose_constraint(
    RgbdFramePtr source_frame,
    RgbdFramePtr reference_frame,
    const fusion::IntrinsicMatrix &cam_param,
    const Sophus::SE3f &pose)
{
    if (source_frame == NULL || reference_frame == NULL)
        return;

    const auto &fx = cam_param.fx;
    const auto &fy = cam_param.fy;
    const auto &cx = cam_param.cx;
    const auto &cy = cam_param.cy;
    const auto &cols = cam_param.width;
    const auto &rows = cam_param.height;
    auto poseInvRef = pose.inverse();

    std::vector<cv::DMatch> matches;

    for (int i = 0; i < source_frame->key_points.size(); ++i)
    {
        const auto &desc_src = source_frame->descriptors.row(i);
        auto pt_in_ref = poseInvRef * source_frame->key_points[i]->pos;
        auto x = fx * pt_in_ref(0) / pt_in_ref(2) + cx;
        auto y = fy * pt_in_ref(1) / pt_in_ref(2) + cy;

        auto th_dist = 0.1f;
        auto min_dist = 64;
        int best_idx = -1;

        if (x >= 0 && y >= 0 && x < cols - 1 && y < rows - 1)
        {
            for (int j = 0; j < reference_frame->key_points.size(); ++j)
            {
                if (reference_frame->key_points[j] == NULL)
                    continue;

                auto dist = (reference_frame->key_points[j]->pos - source_frame->key_points[i]->pos).norm();

                if (dist < th_dist)
                {
                    const auto &desc_ref = reference_frame->descriptors.row(j);
                    auto desc_dist = cv::norm(desc_src, desc_ref, cv::NormTypes::NORM_HAMMING);
                    if (desc_dist < min_dist)
                    {
                        min_dist = desc_dist;
                        best_idx = j;
                    }
                }
            }
        }

        if (best_idx >= 0)
        {
            cv::DMatch match;
            match.queryIdx = i;
            match.trainIdx = best_idx;
            matches.push_back(std::move(match));
        }
    }

    std::vector<cv::DMatch> refined_matches;
    for (int i = 0; i < matches.size(); ++i)
    {

        const auto &match = matches[i];
        const auto query_id = match.queryIdx;
        const auto train_id = match.trainIdx;

        // if (source_frame->cv_key_points[query_id].response > reference_frame->cv_key_points[train_id].response)
        // {
        //     source_frame->key_points[query_id]->observations += reference_frame->key_points[train_id]->observations;
        //     reference_frame->key_points[train_id] = source_frame->key_points[query_id];
        //     reference_frame->cv_key_points[train_id].response = source_frame->cv_key_points[query_id].response;
        // }
        // else
        // {
        //     reference_frame->key_points[train_id]->observations += source_frame->key_points[query_id]->observations;
        //     source_frame->key_points[query_id] = reference_frame->key_points[train_id];
        //     source_frame->cv_key_points[query_id].response = reference_frame->cv_key_points[train_id].response;
        // }

        refined_matches.push_back(std::move(match));
    }

    // cv::Mat outImg;
    // cv::Mat src_image = source_frame->image;
    // cv::Mat ref_image = reference_frame->image;
    // cv::drawMatches(src_image, source_frame->cv_key_points,
    //                 ref_image, reference_frame->cv_key_points,
    //                 refined_matches, outImg, cv::Scalar(0, 255, 0));
    // cv::cvtColor(outImg, outImg, cv::COLOR_BGR2RGB);
    // cv::imshow("correspInit", outImg);
    // cv::waitKey(1);
}

} // namespace fusion
