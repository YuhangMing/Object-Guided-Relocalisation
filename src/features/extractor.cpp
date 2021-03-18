#include "features/extractor.h"

namespace fusion
{

FeatureExtractor::FeatureExtractor()
{
    BRISK = cv::BRISK::create();
    SURF = cv::xfeatures2d::SURF::create();
    cudaORB = cv::cuda::ORB::create();
}

void FeatureExtractor::extract_features_surf(
    const cv::Mat image,
    std::vector<cv::KeyPoint> &keypoints,
    cv::Mat &descriptors)
{
    SURF->detect(image, keypoints);
    BRISK->compute(image, keypoints, descriptors);
}

std::thread FeatureExtractor::extract_features_surf_async(
    const cv::Mat image,
    std::vector<cv::KeyPoint> &keypoints,
    cv::Mat &descriptors)
{
    return std::thread(&FeatureExtractor::extract_features_surf, this, image, std::ref(keypoints), std::ref(descriptors));
}

static cv::Vec4f interpolate_bilinear(cv::Mat map, float x, float y)
{
    int u = (int)std::round(x);
    int v = (int)std::round(y);
    if (u >= 0 && v >= 0 && u < map.cols && v < map.rows)
    {
        return map.ptr<cv::Vec4f>(v)[u];
    }
}

void FeatureExtractor::compute_3d_points(
    const cv::Mat vmap, const cv::Mat nmap,
    const std::vector<cv::KeyPoint> &rawKeypoints,
    const cv::Mat &rawDescriptors,
    std::vector<cv::KeyPoint> &refinedKeypoints,
    cv::Mat &refinedDescriptors,
    std::vector<std::shared_ptr<Point3d>> &mapPoints,
    const Sophus::SE3f Tf2w)
{
    mapPoints.clear();
    refinedDescriptors.release();
    refinedKeypoints.clear();

    const auto rotation = Tf2w.so3();

    if (!vmap.empty())
    {
        auto ibegin = rawKeypoints.begin();
        auto iend = rawKeypoints.end();
        for (auto iter = ibegin; iter != iend; ++iter)
        {
            const auto &x = iter->pt.x;
            const auto &y = iter->pt.y;

            // extract vertex and normal
            cv::Vec4f z = interpolate_bilinear(vmap, x, y);
            cv::Vec4f n = interpolate_bilinear(nmap, x, y);

            if (n(3) > 0 && z(3) > 0 && z == z && n == n)
            {
                std::shared_ptr<Point3d> point(new Point3d());
                point->pos << z(0), z(1), z(2);
                // convert point to world coordinate
                point->observations = 1;
                point->pos = Tf2w * point->pos;
                point->vec_normal << n(0), n(1), n(2);
                point->vec_normal = rotation * point->vec_normal;
                point->descriptors = rawDescriptors.row(std::distance(rawKeypoints.begin(), iter));
                refinedDescriptors.push_back(point->descriptors);
                refinedKeypoints.push_back(std::move(*iter));
                mapPoints.push_back(std::move(point));
            }
        }
    }
}

} // namespace fusion
