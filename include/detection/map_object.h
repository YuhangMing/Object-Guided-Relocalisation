#ifndef SLAM_MAP_OBJECT_H
#define SLAM_MAP_OBJECT_H

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include "detection/map_cuboid.h"
// #include "detection/map_quadric.h"

namespace fusion
{

struct Object3d
{
    int label;
    int observation_count;

    // index, centroid and covariance of the primary cuboid
    int primary_cuboid_idx;
    Eigen::Vector3d pos;
    Eigen::Matrix3d cov;
    
    // all possible cuboids
    // if it's frame object, it should have size=1
    std::vector<std::shared_ptr<Cuboid3d>> v_all_cuboids;

    Object3d(std::shared_ptr<Cuboid3d> cub);
    Object3d(std::shared_ptr<Object3d> ref_obj);
    void copyFrom(std::shared_ptr<Object3d> ref_obj);
    
    int cuboid_idx; // cuboid matches best with new observation
    // bool bbox3d_overlap(Eigen::Vector3d &cent, 
    //                     std::vector<float> &dim, 
    //                     float s);
    float bbox3d_overlap(Eigen::Vector3d &cent, 
                        std::vector<float> &dim, 
                        float s);
    void update_object(std::shared_ptr<Cuboid3d> obs);
    void update_cuboid(std::shared_ptr<Cuboid3d> cub1, std::shared_ptr<Cuboid3d> cub2);

    Object3d(std::string file_name, int& start_line);
    void writeToFile(std::string file_name);
    void readFromFile(std::string file_name, int& start_line);
    
};

// inline Object3d::Object3d(std::shared_ptr<Cuboid3d> cub)
// {
//     label = cub->label;
//     observation_count = cub->observation;
//     cuboid_idx = 0;
//     v_all_cuboids.push_back(std::move(cub));
// }

} // namespace fusion

#endif