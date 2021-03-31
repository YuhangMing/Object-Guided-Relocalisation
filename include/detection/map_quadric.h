#ifndef SLAM_MAP_QUADRIC_H
#define SLAM_MAP_QUADRIC_H

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

struct Quadric3d
{
    // bool visited;
    // Eigen::Vector3f pos;
    // Eigen::Vector3f vec_normal;
    // size_t observations;
    // cv::Mat descriptors;

	int label;
	float confidence;
	size_t observation;
    float orientation[3];       // angles
    float translation[3];       // 
    float principal_axes[3];    //
    Eigen::Matrix4f Q, T, Q_ori, Q_dual;

    // inline void 
};

#endif