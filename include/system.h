#ifndef SYSTEM_H
#define SYSTEM_H

#include <thread>
#include <eigen3/Eigen/Core>
#include <opencv2/opencv.hpp>
#include "data_struct/intrinsic_matrix.h"
#include "data_struct/rgbd_frame.h"
#include "voxel_hashing/voxel_hashing.h"
#include "tracking/rgbd_odometry.h"
#include "relocalization/relocalizer.h"
#include "keyframe_graph/keyframe_graph.h"
#include "features/extractor.h"
#include "features/matcher.h"
#include "map_manager.h"
#include "detection/detector.h"

namespace fusion
{

class SubMapManager;
class DenseOdometry;
// class MaskRCNN;

class System
{
public:
    ~System();
    System(const fusion::IntrinsicMatrix base, const int NUM_PYR, bool bSemantic=true, bool bLoadSMap=false);
    void process_images(const cv::Mat depth, const cv::Mat image, const fusion::IntrinsicMatrix base, 
                        bool bSubmapping, bool bSemantic, bool bRecordSequence);
    // pure relocalization
    void set_frame_id(size_t sid);
    void relocalize_image(const cv::Mat depth, const cv::Mat image, const fusion::IntrinsicMatrix base);
    void relocalization();

    // get rendered ray tracing map
    cv::Mat get_detected_image();
    cv::Mat get_shaded_depth();
    cv::Mat get_rendered_scene() const;
    cv::Mat get_rendered_scene_textured() const;
    // detection result
    cv::Mat get_NOCS_map() const;
    cv::Mat get_segmented_mask() const;

    // create a mesh from the map
    // and save it to a named file
    // it only contains vertex data
    void save_mesh_to_file(const char *str);

    // create mesh and store in the address
    // users are reponsible for allocating
    // the adresses in CUDA using `cudaMalloc`
    size_t fetch_mesh_vertex_only(float *vertex);
    size_t fetch_mesh_with_normal(float *vertex, float *normal);
    size_t fetch_mesh_with_colour(float *vertex, unsigned char *colour);

    // key points
    void fetch_key_points(float *points, size_t &count, size_t max);
    void fetch_key_points_with_normal(float *points, float *normal, size_t &max_size);

    bool is_initialized;
    mutable bool b_reloc_attp;

    // system controls
    void change_colour_mode(int colour_mode = 0);
    void change_run_mode(int run_mode = 0);
    void restart();
    void setLost(bool lost);

    void writeMapToDisk(std::string file_name) const;
    void readMapFromDisk(std::string file_name);
    // void recordSequence(std::string dir) const;

    // visualization
    Eigen::Matrix4f get_camera_pose() const;
    std::vector<Eigen::Matrix<float, 4, 4>> getKeyFramePoses() const;
    std::vector<Eigen::Matrix<float, 4, 4>> getGTposes() const;
    std::vector<Eigen::Matrix<float, 4, 4>> vRelocPoses;
    std::vector<Eigen::Matrix<float, 4, 4>> getRelocPoses() const;
    std::vector<Eigen::Matrix<float, 4, 4>> vRelocPosesGT;
    std::vector<Eigen::Matrix<float, 4, 4>> getRelocPosesGT() const;
    
    int get_num_objs() const;
    int get_reloc_num_objs() const;
    std::vector<std::pair<int, std::vector<float>>> get_objects(bool bMain) const;
    std::vector<std::pair<int, std::vector<float>>> get_object_cuboids() const;
    std::vector<std::pair<int, std::vector<float>>> get_reloc_cuboids(int usePose) const;
    std::vector<float> get_obj_centroid_axes(int idx_obj);
    std::vector<float> get_reloc_obj_centroid_axes(int idx_obj, int usePose);
    int get_object_pts(float *points, size_t &count, int idx_obj);
    int get_reloc_obj_pts(float *points, size_t &count, int idx_obj, bool b_useGT);

    std::vector<float> get_plane_normals() const;

    std::string output_file_name;

    // test
    // std::map<int, Eigen::Matrix<float, 4, 4>> maGTposes;
    std::vector<Eigen::Matrix<float, 4, 4>> vGTposes;
    int reloc_frame_id;
    void load_pose_info(std::string folder, int seq_id);
    void load_from_text(std::string file_name, std::vector<Eigen::Matrix4f>& v_results);
    // void load_from_text(std::string file_name, std::map<int, Eigen::Matrix4f>& ma_results);
    // std::map<int, Eigen::Matrix4f> maORBposes;
    // std::map<int, Eigen::Matrix4f> getORBPoseResults() const;
    bool pause_window;

    std::vector<Eigen::Matrix4f> v_NOCSpose_results;
    std::vector<Eigen::Matrix4f> getNOCSPoseResults() const;
    std::vector<Eigen::Matrix4f> v_MaskRCNN_results;
    std::vector<Eigen::Matrix4f> getMaskRCNNResults() const;

private:
    // no more separate keyframe structure, keyframe is frame
    RgbdFramePtr current_frame;
    RgbdFramePtr current_keyframe;

    size_t frame_id;
    // size_t sequence_id;
    size_t frame_start_reloc_id;

    // System modules
    std::shared_ptr<SubMapManager> manager;
    // std::shared_ptr<DenseMapping> mapping;
    std::shared_ptr<DenseOdometry> odometry;
    std::shared_ptr<KeyFrameGraph> graph;
    std::shared_ptr<Relocalizer> relocalizer;
    std::shared_ptr<FeatureExtractor> extractor;
    std::shared_ptr<DescriptorMatcher> matcher;
    // NOCS * nocsdetector;
    semantic::Detector * detector;
    std::thread graphThread;

    // Return TRUE if a new key frame is desired
    // return FALSE otherwise
    // TODO: this needs to be redesigned.
    bool keyframe_needed() const;
    void create_keyframe();
    void initialization();
    bool hasNewKeyFrame;
    Sophus::SE3d initialPose;

    cv::cuda::GpuMat device_depth_float;
    cv::cuda::GpuMat device_image_uchar;
    cv::cuda::GpuMat device_vmap_cast;
    cv::cuda::GpuMat device_nmap_cast;

    int renderIdx;
    std::vector<Sophus::SE3d> gt_pose;

    // semantic analysis
    void extract_objects(RgbdFramePtr frame, bool bGeoSeg, float lamb, float tao, int win_size, int thre);
    void extract_planes(RgbdFramePtr frame);
    void extract_semantics(RgbdFramePtr frame, bool bGeoSeg, float lamb, float tao, int win_size, int thre);
    
};

} // namespace fusion

#endif