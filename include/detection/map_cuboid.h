#ifndef SLAM_MAP_CUBOID_H
#define SLAM_MAP_CUBOID_H

#include <iostream>
#include <vector>
#include <memory>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

namespace fusion{

struct Cuboid3d
{
    // bool visited;
    // Eigen::Vector3f pos;
    // Eigen::Vector3f vec_normal;
    // size_t observations;
    // cv::Mat descriptors;

    // obj class info
	int label;
	float confidence;
    std::vector<float> all_class_confidence;
	int observation;
    // cv::Mat data_pts;   // in float type
    std::vector<Eigen::Vector3f> v_data_pts;

    // cuboid info
    /*
    if tracking good: data_pts and centroid are in World Coord Frame
    if tracking lost: data_pts and centroid are in Cam Coord Frame
    */
    // -----------------
    // !! with MaskRCNN
	// Eigen::Vector3d centroid;
    // std::vector<Eigen::Vector3d> axes;
    // std::vector<float> axes_length;
    // // pose w.r.t. the world origin
    // Eigen::Matrix3d rotation;
    // Eigen::Vector3d translation;        // translation = centroid
    // std::vector<double> rotated_dim;         // blb and trf of cuboid in rotated/aligned coordinate system
    // // in the order of: blb, brb, brf, blf; tlf, tlb, trb, trf
    // std::vector<float> cuboid_corner_pts;   // corner pts in the original world coordinate system
    // std::vector<std::vector<float>> corner_candidates;
    // -----------------
    
    // !! with NOCS
    // parameters for displaying the cuboid (primary cuboid)
    // cv::Mat coord;
    Eigen::Matrix4d pose;       // pose of the object w.r.t. the world origin in the map and the camera center in the relocalized frame
    std::vector<float> dims;    // dimensions of the object, x/y/z axis, in NOCS space
    float scale;                // scale factor that transform coordinates from NOCS to Camera Space
    cv::Mat cuboid_corner_pts;  // corners, axes & centroid are coordinates in the Camera Space
    cv::Mat axes;
    Eigen::Vector3d centroid;
    // -----------------

    // probabilitic model
    // all centroids stored here are from raw detection
    Eigen::Vector3d mean;
    Eigen::Matrix3d cov;  // computed using MLE from measurements
    std::vector<Eigen::Vector3d> vCentroids;    // stores all the centroinds belong to current cuboid
    Eigen::Matrix3d cov_propagated; // propagated back from sensor
    std::vector<double> vScales;
    double sigma_scale;

    // // store the all the TP and FP detection results
    // std::vector<std::vector<float>> v_all_class_confidence;
    // std::vector<int> v_observation;
    // std::vector<Eigen::Matrix4d> v_pose;
    // std::vector<Eigen::Vector3d> v_centroid;
    // std::vector<std::vector<float>> v_dims;
    // std::vector<float> v_scale;
    // std::vector<cv::Mat> v_cuboid_corner_pts;
    // std::vector<cv::Mat> v_axes;
    // std::vector<Eigen::Vector3d> v_mean;
    // std::vector<Eigen::Matrix3d> v_covMLE;
    // std::vector<std::vector<Eigen::Vector3d>> v_vCentroids;
    // std::vector<Eigen::Matrix3d> v_covPropagated;

    Cuboid3d(int label, float confidence);
    Cuboid3d(int l, float c, Eigen::Matrix4d Rt, std::vector<float> d, float s, Eigen::Matrix3d Sigma_t);
    Cuboid3d(std::shared_ptr<Cuboid3d> ref_cub);
    void copyFrom(std::shared_ptr<Cuboid3d> ref_cub);
    void add_points(std::vector<Eigen::Vector3f> points);
    int find_axis_correspondence(std::vector<Eigen::Vector3f> plane_normals, int& sign);
    void align_with_palne_normal(Eigen::Vector3f gp_normal);
    void find_bounding_cuboid();
    void update_confidence(std::vector<float> obs_confidence);
    
    Cuboid3d(std::string file_name, int start_line);
    void writeToFile(std::string file_name);
    void readFromFile(std::string file_name, int start_line);

    // std::vector<float> find_centroid();
    // void find_principal_axes();
    // bool find_bounding_cuboid();
    // void merge_cuboids(std::shared_ptr<Cuboid3d> pNewCuboid);
    // bool is_overlapped(std::shared_ptr<Cuboid3d> pNewCuboid);
    // void update_label(std::shared_ptr<Cuboid3d> pNewCuboid);
    // void fit_cuboid(std::shared_ptr<Cuboid3d> pTargetCuboid, int mask_width);

    // inline Cuboid3d();
    // inline void add_point(Eigen::Vector3f point);
    // inline void find_principal_axes();

private:
    Eigen::Matrix3f calculate_rotation(Eigen::Vector3f current, Eigen::Vector3f target);

    // bool align_principal_axes();
    // void translate_corner_pts(std::shared_ptr<Cuboid3d> pTargetCuboid);

};

// inline Cuboid3d::Cuboid3d() : axes(3), axes_length(3) {}

// inline void Cuboid3d::add_point(Eigen::Vector3f point){
//     cv::Mat cvMat(1, 3, CV_32F);
//     cvMat.at<float>(0,0) = point(0);
//     cvMat.at<float>(0,1) = point(1);
//     cvMat.at<float>(0,2) = point(2);
//     data_pts.push_back(cvMat);
// }

// inline void Cuboid3d::find_principal_axes(){
//     cv::PCA pca_analysis(data_pts, cv::Mat(), cv::PCA::DATA_AS_ROW);
//     // std::cout << "Eigenvectors are:\n" << pca_analysis.eigenvectors << std::endl;
//     // std::cout << "Eigenvalues are:\n" << pca_analysis.eigenvalues << std::endl;
//     // std::cout << "MEANS are:\n" << pca_analysis.mean << std::endl;
//     for(size_t i=0; i<3; ++i){
//         // centroid
//         centroid[i] = pca_analysis.mean.at<float>(i);
//         // principal axes
//         Eigen::Vector3f point;
//         point << pca_analysis.eigenvectors.at<float>(i, 0),
//                     pca_analysis.eigenvectors.at<float>(i, 1),
//                     pca_analysis.eigenvectors.at<float>(i, 2);
//         axes[i] = point;
//         axes_length[i] = pca_analysis.eigenvalues.at<float>(i);
//     }
//     // std::cout << "Axes are: \n"
//     //             << axes[0](0) << ", " << axes[0](1) << ", " << axes[0](2) << "; \n"
//     //             << axes[1](0) << ", " << axes[1](1) << ", " << axes[1](2) << "; \n"
//     //             << axes[2](0) << ", " << axes[2](1) << ", " << axes[2](2) << std::endl;
//     // std::cout << "Axes' lengthes are: \n"
//     //             << axes_length[0] << ", "
//     //             << axes_length[1] << ", "
//     //             << axes_length[2] << std::endl;
//     // std::cout << "Centroid is: \n"
//     //             << centroid[0] << ", " << centroid[1] << ", " << centroid[2] << std::endl;
// }

}   // namespace fusion

#endif