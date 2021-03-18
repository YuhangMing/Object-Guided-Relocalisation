#include <fstream>
#include "input/tum_loader.h"

namespace fusion
{

TUMLoader::TUMLoader(const std::string &root_path)
    : root_path(root_path), image_id(0), images_loaded(0)
{
    load_association();
}

void TUMLoader::get_next_images(cv::Mat &depth, cv::Mat &image)
{
    if (image_id >= images_loaded)
        return;

    std::string fullpath_image = root_path + image_filenames[image_id];
    std::string fullpath_depth = root_path + depth_filenames[image_id];

    image = cv::imread(fullpath_image, cv::IMREAD_UNCHANGED);
    depth = cv::imread(fullpath_depth, cv::IMREAD_UNCHANGED);

    image_id += 1;
}

bool TUMLoader::has_images() noexcept
{
    return image_id < images_loaded;
}

bool TUMLoader::load_association()
{
    std::ifstream file(root_path + "association.txt");

    if (!file.is_open())
    {
        std::cout << "Cannot locate the association file.\n";
        return false;
    }

    double ts;
    std::string filename_depth;
    std::string filename_image;

    while (file >> ts >> filename_depth >> ts >> filename_image)
    {
        image_filenames.push_back(filename_image);
        depth_filenames.push_back(filename_depth);
        time_stamps.push_back(ts);
        images_loaded += 1;
    }

    if (images_loaded == 0)
    {
        std::cout << "TUM Loading failed.\n";
        return false;
    }
    else
    {
        std::cout << "Total of " << images_loaded << " Images Loaded.\n";
        return true;
    }
}

} // namespace fusion