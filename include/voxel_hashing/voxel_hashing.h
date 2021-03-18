#ifndef DENSE_MAPPING_H
#define DENSE_MAPPING_H

#include <memory>
#include "data_struct/map_struct.h"
#include "data_struct/rgbd_frame.h"
#include "data_struct/map_object.h"
#include "data_struct/map_cuboid.h"
#include "data_struct/map_quadric.h"
#include "tracking/device_image.h"

namespace fusion
{

class DenseMapping
{
public:
  ~DenseMapping();
  DenseMapping(const fusion::IntrinsicMatrix &K, int idx, bool bTrack, bool bRender);

  void update(RgbdImagePtr frame);
  void update(cv::cuda::GpuMat depth, cv::cuda::GpuMat image, const Sophus::SE3d pose);
  void raycast(cv::cuda::GpuMat &vmap, cv::cuda::GpuMat &image, const Sophus::SE3d pose);
  void raycast(cv::cuda::GpuMat &vmap, cv::cuda::GpuMat &image, cv::cuda::GpuMat &mask, const Sophus::SE3d pose);
  void create_blocks(RgbdImagePtr frame);

  void raycast_check_visibility(
      cv::cuda::GpuMat &vmap,
      cv::cuda::GpuMat &image,
      const Sophus::SE3d pose);

  void reset_mapping();

  size_t fetch_mesh_vertex_only(void *vertex);
  size_t fetch_mesh_with_normal(void *vertex, void *normal);
  size_t fetch_mesh_with_colour(void *vertex, void *normal);

  void writeMapToDisk(std::string file_name);
  void readMapFromDisk(std::string file_name);

  // submap data structures
  int submapIdx;
  bool bTrack, bRender;
  float visible_percent;
  Sophus::SE3d poseGlobal;
  std::vector<RgbdFramePtr> vKFs; // stored whenever a new kf is created
  std::vector<Eigen::Matrix4f> vKFposes;  // only pose stored, used in load map and relocalize only
  float CheckVisPercent();
  void check_visibility(RgbdImagePtr frame);

  // semantic
  void color_objects(RgbdImagePtr frame);
  // plane
  std::vector<Eigen::Vector3f> plane_normals;
  void update_planes(RgbdFramePtr frame);
  // object
  std::vector<std::shared_ptr<Object3d>> v_objects;                         // primary object, used for relocalisation
  std::map<int, std::vector<std::shared_ptr<Object3d>>> object_dictionary;  // back-up dictionary, mainly used in map construction
  void update_objects(RgbdFramePtr frame);
  // // cuboid
  // std::vector<std::shared_ptr<Cuboid3d>> object_cuboids;  // primary objects
  // std::map<int, std::vector<std::shared_ptr<Cuboid3d>>> cuboid_dictionary;  // back-up dictionary
  // void update_cuboids(RgbdFramePtr frame);
  // void estimate_cuboids(RgbdImagePtr frame);                      // old-version GPU
  // void estimate_cuboids(RgbdFramePtr frame, bool tracking_lost);  // old-version CPU
  // // quadric
  // std::vector<std::shared_ptr<Quadric3d>> object_quadrics;
  // void estimate_quadrics(RgbdFramePtr frame, bool tracking_lost);

private:
  IntrinsicMatrix cam_params;
  MapStruct<true> device_map;

  // for map udate
  cv::cuda::GpuMat flag;
  cv::cuda::GpuMat pos_array;
  uint count_visible_block;
  HashEntry *visible_blocks;

  // for raycast
  cv::cuda::GpuMat zrange_x;
  cv::cuda::GpuMat zrange_y;
  uint count_rendering_block;
  RenderingBlock *rendering_blocks;

  // // for semantic
  // void update_one_cuboid(std::shared_ptr<Cuboid3d> map, 
  //                        std::shared_ptr<Cuboid3d> obs,
  //                        bool update_conf = true);
  // void calculate_overlap(Eigen::Vector3d& cent1, std::vector<float>& dim1, float s1,
  //                        Eigen::Vector3d& cent2, std::vector<float>& dim2, float s2,
  //                        float& rMap, float& rFrame);
  // cv::cuda::GpuMat cuboid;
};

} // namespace fusion

#endif