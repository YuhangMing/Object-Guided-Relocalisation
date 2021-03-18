#ifndef SLAM_MAP_POINT_H
#define SLAM_MAP_POINT_H

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

struct Point3d
{
    bool visited;
    Eigen::Vector3f pos;
    Eigen::Vector3f vec_normal;
    size_t observations;
    cv::Mat descriptors;
};

#endif