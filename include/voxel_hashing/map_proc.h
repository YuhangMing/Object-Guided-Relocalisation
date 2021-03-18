#ifndef FUSION_VOXEL_HASHING_MAP_PROC
#define FUSION_VOXEL_HASHING_MAP_PROC

#include <sophus/se3.hpp>
#include <opencv2/cudaarithm.hpp>
#include "data_struct/map_struct.h"
#include "data_struct/intrinsic_matrix.h"

namespace fusion
{
namespace cuda
{

void update(
    MapStorage map_struct,
    MapState state,
    const cv::cuda::GpuMat depth,
    const cv::cuda::GpuMat image,
    const Sophus::SE3d &frame_pose,
    const IntrinsicMatrix K,
    cv::cuda::GpuMat &cv_flag,
    cv::cuda::GpuMat &cv_pos_array,
    HashEntry *visible_blocks,
    uint &visible_block_count);

void update_weighted(
    MapStorage map_struct,
    MapState state,
    const cv::cuda::GpuMat depth,
    const cv::cuda::GpuMat normal,
    const cv::cuda::GpuMat image,
    const Sophus::SE3d &frame_pose,
    const IntrinsicMatrix K,
    cv::cuda::GpuMat &cv_flag,
    cv::cuda::GpuMat &cv_pos_array,
    HashEntry *visible_blocks,
    uint &visible_block_count);

void check_visibility(
    MapStorage map_struct,
    MapState state,
    const cv::cuda::GpuMat depth,
    const Sophus::SE3d &frame_pose,
    const IntrinsicMatrix K,
    cv::cuda::GpuMat &cv_flag,
    cv::cuda::GpuMat &cv_pos_array,
    HashEntry *visible_blocks,
    uint &visible_block_count);

void color_objects(
    MapStorage map_struct,
    MapState state,
    const cv::cuda::GpuMat depth,
    const cv::cuda::GpuMat image,
    const cv::cuda::GpuMat mask,
    const Sophus::SE3d &frame_pose,
    const IntrinsicMatrix K,
    cv::cuda::GpuMat &cv_flag,
    cv::cuda::GpuMat &cv_pos_array,
    HashEntry *visible_blocks,
    uint &visible_block_count);

// void estimate_cuboids(
//     const cv::cuda::GpuMat vmap,
//     const cv::cuda::GpuMat mask,
//     const cv::cuda::GpuMat bbox,
//     const unsigned char label,
//     cv::cuda::GpuMat &cuboid);

void create_rendering_blocks(
    uint count_visible_block,
    uint &count_redering_block,
    HashEntry *visible_blocks,
    cv::cuda::GpuMat &zrange_x,
    cv::cuda::GpuMat &zrange_y,
    RenderingBlock *rendering_blocks,
    const Sophus::SE3d &frame_pose,
    const IntrinsicMatrix cam_params);

void raycast(
    MapStorage map_struct,
    MapState state,
    cv::cuda::GpuMat vmap,
    cv::cuda::GpuMat nmap,
    cv::cuda::GpuMat zrange_x,
    cv::cuda::GpuMat zrange_y,
    const Sophus::SE3d &pose,
    const IntrinsicMatrix intrinsic_matrix);

void raycast_with_colour(
    MapStorage map_struct,
    MapState state,
    cv::cuda::GpuMat vmap,
    cv::cuda::GpuMat nmap,
    cv::cuda::GpuMat image,
    cv::cuda::GpuMat zrange_x,
    cv::cuda::GpuMat zrange_y,
    const Sophus::SE3d &pose,
    const IntrinsicMatrix intrinsic_matrix);
void raycast_with_object(
    MapStorage map_struct,
    MapState state,
    cv::cuda::GpuMat vmap,
    cv::cuda::GpuMat nmap,
    cv::cuda::GpuMat image,
    cv::cuda::GpuMat mask,
    cv::cuda::GpuMat zrange_x,
    cv::cuda::GpuMat zrange_y,
    const Sophus::SE3d &pose,
    const IntrinsicMatrix intrinsic_matrix);

void create_mesh_vertex_only(
    MapStorage map_struct,
    MapState state,
    uint &block_count,
    HashEntry *block_list,
    uint &triangle_count,
    void *vertex_data);

void create_mesh_with_normal(
    MapStorage map_struct,
    MapState state,
    uint &block_count,
    HashEntry *block_list,
    uint &triangle_count,
    void *vertex_data,
    void *vertex_normal);

void create_mesh_with_colour(
    MapStorage map_struct,
    MapState state,
    uint &block_count,
    HashEntry *block_list,
    uint &triangle_count,
    void *vertex_data,
    void *vertex_colour);

void count_visible_entry(
    const MapStorage map_struct,
    const MapSize map_size,
    const IntrinsicMatrix &K,
    const Sophus::SE3d frame_pose,
    HashEntry *const visible_entry,
    uint &visible_block_count);

void create_new_block(
    MapStorage map_struct,
    MapState state,
    const cv::cuda::GpuMat depth,
    const Sophus::SE3d &frame_pose,
    const IntrinsicMatrix K,
    cv::cuda::GpuMat &cv_flag,
    cv::cuda::GpuMat &cv_pos_array,
    HashEntry *visible_blocks,
    uint &visible_block_count
);

} // namespace cuda
} // namespace fusion

#endif