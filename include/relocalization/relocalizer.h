#ifndef SLAM_RELOCALIZER_H
#define SLAM_RELOCALIZER_H

#include "data_struct/rgbd_frame.h"
#include "data_struct/map_point.h"
#include "data_struct/map_cuboid.h"
#include "data_struct/map_object.h"

namespace fusion
{

class Relocalizer
{
public:
    Relocalizer(const Eigen::Matrix3f intrinsic_inv);
    void set_target_frame(std::shared_ptr<RgbdFrame> frame);
    void compute_pose_candidates(std::vector<Sophus::SE3d> &candidates);

    // Multiple instances of the same class
    std::vector<Eigen::Matrix4d> object_data_association(std::vector<std::shared_ptr<Object3d>> frame_obj, 
                                                 std::vector<std::shared_ptr<Object3d>> map_obj,
                                                 std::vector<std::vector< std::pair<int, std::pair<int, int>> >>& vv_inlier_pairs,
                                                 bool & b_enough_corresp, bool & b_recovered);
    
    std::vector<Eigen::Matrix4d> object_relocalization(std::vector<std::shared_ptr<Object3d>> frame_obj, 
                                                 std::vector<std::shared_ptr<Object3d>> map_obj,
                                                 std::vector<std::pair<int, int>>& v_best_map_cub_labidx,
                                                 bool & b_enough_corresp, bool & b_recovered);
    // Only one instance per class assumed.
    // Almost the same as object_relocalization except for the finding combination part
    std::vector<Eigen::Matrix4d> object_guided_relocalization(std::vector<std::shared_ptr<Object3d>> frame_obj, 
                                                 std::vector<std::shared_ptr<Object3d>> map_obj,
                                                 std::vector<std::pair<int, int>>& v_best_map_cub_labidx,
                                                 bool & b_enough_corresp, bool & b_recovered);
    
    void set_maWeights(std::map<int, double> new_weights);

    // Other relocalizaiton process tested
    Eigen::Matrix4d NOCS_relocalization(std::vector<std::shared_ptr<Cuboid3d>> frame_obj, 
                                        std::vector<std::shared_ptr<Cuboid3d>> map_obj,
                                        bool & b_enough_corresp, bool & b_recovered);
    Eigen::Matrix4d pose_average(std::vector<Eigen::Quaterniond> vRot, 
                                 std::vector<Eigen::Vector3d> vTrans,
                                 std::vector<double> weights);
    Eigen::Matrix4d pose_vec_relocalization(std::vector<std::shared_ptr<Cuboid3d>> frame_obj, 
                                            std::vector<std::shared_ptr<Cuboid3d>> map_obj,
                                            bool & b_enough_corresp, bool & b_recovered);
    Eigen::Matrix4d compute_pose_from_poses(Eigen::Matrix4d pose_frame, Eigen::Matrix4d pose_obj);
    Eigen::Matrix4d semantic_relocalization(std::vector<std::shared_ptr<Cuboid3d>> frame_obj, 
                                            std::vector<std::shared_ptr<Cuboid3d>> map_obj,
                                            bool b_use_corners,
                                            Eigen::Matrix4d gtPose,
                                            bool & b_enough_corresp, bool & b_recovered);
    Eigen::Matrix4d semantic_reloc_with_dic(std::vector<std::shared_ptr<Cuboid3d>> frame_obj, 
                                            std::map<int, std::vector<std::shared_ptr<Cuboid3d>>> map_obj_dic,
                                            bool b_use_corners,
                                            Eigen::Matrix4d gtPose,
                                            bool & b_enough_corresp, bool & b_recovered);

    // Relocalization with map recognition
    Eigen::Matrix4d semantic_reloc_recog(std::vector<std::shared_ptr<Cuboid3d>> frame_obj, 
                                         std::vector<std::vector<std::shared_ptr<Cuboid3d>>> maps,
                                         bool b_use_corners,
                                         bool & b_enough_corresp, bool & b_recovered, int & mapId);
private:
    std::vector<int> label_pose, label_vec;

    std::shared_ptr<RgbdFrame> target_frame;
    Eigen::Matrix3f KInv;

    // Initialize a Map of string & vector of int using initializer_list
    // this weight represents the relative size of objects
    // smaller value tends to be inlier
    // std::map<int, double> maWeights = 	{
    //                             { 1, 0.9},
    //                             { 2, 0.9},
    //                             { 3, 0.8},
    //                             { 4, 1.},
    //                             { 5, 0.7},
    //                             { 6, 1.}
    //                             };
    std::map<int, double> maWeights = 	{
                                { 1, 1.},
                                { 2, 1.},
                                { 3, 1.},
                                { 4, 1.},
                                { 5, 1.},
                                { 6, 1.}
                                };
    std::map<int, std::string> maLableText = {
                                {1, "bottle"},
                                {2, "bowl"},
                                {3, "camera"},
                                {4, "can"},
                                {5, "laptop"},
                                {6, "mug"}
                                };
};

} // namespace fusion

#endif