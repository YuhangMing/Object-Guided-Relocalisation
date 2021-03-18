#ifndef FUSION_UTILS_TUM_LOADER_H
#define FUSION_UTILS_TUM_LOADER_H

#include <string>
#include <opencv2/opencv.hpp>

namespace fusion
{

class TUMLoader
{
public:
    TUMLoader(const std::string &root_path);
    void get_next_images(cv::Mat &depth, cv::Mat &image);
    bool has_images() noexcept;

private:
    std::string root_path;
    bool load_association();

    size_t image_id;
    size_t images_loaded;

    std::vector<double> time_stamps;
    std::vector<std::string> depth_filenames;
    std::vector<std::string> image_filenames;
};

} // namespace fusion

#endif