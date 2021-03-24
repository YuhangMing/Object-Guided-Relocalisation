#include "voxel_hashing/map_proc.h"
#include "math/matrix_type.h"
#include "math/vector_type.h"
#include "utils/safe_call.h"
#include "macros.h"
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <thrust/device_vector.h>

namespace fusion
{
namespace cuda
{

FUSION_DEVICE inline bool is_vertex_visible(
    Vector3f pt, Matrix3x4f inv_pose,
    int cols, int rows, float fx,
    float fy, float cx, float cy)
{
    pt = inv_pose(pt);
    Vector2f pt2d = Vector2f(fx * pt.x / pt.z + cx, fy * pt.y / pt.z + cy);
    return !(pt2d.x < 0 || pt2d.y < 0 ||
             pt2d.x > cols - 1 || pt2d.y > rows - 1 ||
             pt.z < param.zmin_update || pt.z > param.zmax_update);
}

FUSION_DEVICE inline bool is_block_visible(
    const Vector3i &block_pos,
    Matrix3x4f inv_pose,
    int cols, int rows, float fx,
    float fy, float cx, float cy)
{
    float scale = param.block_size_metric();
#pragma unroll
    for (int corner = 0; corner < 8; ++corner)
    {
        Vector3i tmp = block_pos;
        tmp.x += (corner & 1) ? 1 : 0;
        tmp.y += (corner & 2) ? 1 : 0;
        tmp.z += (corner & 4) ? 1 : 0;

        if (is_vertex_visible(tmp * scale, inv_pose, cols, rows, fx, fy, cx, cy))
            return true;
    }

    return false;
}

__global__ void check_visibility_flag_kernel(
    MapStorage map_struct, uchar *flag, Matrix3x4f inv_pose,
    int cols, int rows, float fx, float fy, float cx, float cy)
{
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= param.num_total_hash_entries_)
        return;

    HashEntry &current = map_struct.hash_table_[idx];
    if (current.ptr_ != -1)
    {
        switch (flag[idx])
        {
        default:
        {
            if (is_block_visible(current.pos_, inv_pose, cols, rows, fx, fy, cx, cy))
            {
                flag[idx] = 1;
            }
            else
            {
                // map_struct.delete_block(current);
                flag[idx] = 0;
            }

            return;
        }
        case 2:
        {
            flag[idx] = 1;
            return;
        }
        }
    }
}

__global__ void copy_visible_block_kernel(HashEntry *hash_table, HashEntry *visible_block, const uchar *flag, const int *pos)
{
    // printf("Block %d thread %d, ", blockIdx.x, threadIdx.x);

    
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= param.num_total_hash_entries_)
        return;

    if (flag[idx] == 1 && pos[idx] < param.num_total_hash_entries_)
        visible_block[pos[idx]] = hash_table[idx];
}

FUSION_DEVICE inline Vector2f project(
    Vector3f pt, float fx, float fy, float cx, float cy)
{
    return Vector2f(fx * pt.x / pt.z + cx, fy * pt.y / pt.z + cy);
}

FUSION_DEVICE inline Vector3f unproject(
    int x, int y, float z, float invfx, float invfy, float cx, float cy)
{
    return Vector3f(invfx * (x - cx) * z, invfy * (y - cy) * z, z);
}

FUSION_DEVICE inline Vector3f unproject_world(
    int x, int y, float z, float invfx,
    float invfy, float cx, float cy, Matrix3x4f pose)
{
    return pose(unproject(x, y, z, invfx, invfy, cx, cy));
}

FUSION_DEVICE inline int create_block(MapStorage &map_struct, const Vector3i block_pos)
{
    int hash_index;
    createBlock(map_struct, block_pos, hash_index);
    return hash_index;
}

__global__ void create_blocks_kernel(MapStorage map_struct, cv::cuda::PtrStepSz<float> depth,
                                     float invfx, float invfy, float cx, float cy,
                                     Matrix3x4f pose, uchar *flag)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= depth.cols || y >= depth.rows)
        return;

    float z = depth.ptr(y)[x];
    if (isnan(z) || z < param.zmin_update || z > param.zmax_update)
        return;

    float z_thresh = param.truncation_dist() * 0.5;
    float z_near = max(param.zmin_update, z - z_thresh);
    float z_far = min(param.zmax_update, z + z_thresh);
    if (z_near >= z_far)
        return;

    Vector3i block_near = voxelPosToBlockPos(worldPtToVoxelPos(unproject_world(x, y, z_near, invfx, invfy, cx, cy, pose), param.voxel_size));
    Vector3i block_far = voxelPosToBlockPos(worldPtToVoxelPos(unproject_world(x, y, z_far, invfx, invfy, cx, cy, pose), param.voxel_size));

    Vector3i d = block_far - block_near;
    Vector3i increment = Vector3i(d.x < 0 ? -1 : 1, d.y < 0 ? -1 : 1, d.z < 0 ? -1 : 1);
    Vector3i incre_abs = Vector3i(abs(d.x), abs(d.y), abs(d.z));
    Vector3i incre_err = Vector3i(incre_abs.x << 1, incre_abs.y << 1, incre_abs.z << 1);

    int err_1;
    int err_2;

    // Bresenham's line algorithm
    // details see : https://en.m.wikipedia.org/wiki/Bresenham%27s_line_algorithm
    if ((incre_abs.x >= incre_abs.y) && (incre_abs.x >= incre_abs.z))
    {
        err_1 = incre_err.y - 1;
        err_2 = incre_err.z - 1;
        flag[create_block(map_struct, block_near)] = 2;
        for (int i = 0; i < incre_abs.x; ++i)
        {
            if (err_1 > 0)
            {
                block_near.y += increment.y;
                err_1 -= incre_err.x;
            }

            if (err_2 > 0)
            {
                block_near.z += increment.z;
                err_2 -= incre_err.x;
            }

            err_1 += incre_err.y;
            err_2 += incre_err.z;
            block_near.x += increment.x;
            flag[create_block(map_struct, block_near)] = 2;
        }
    }
    else if ((incre_abs.y >= incre_abs.x) && (incre_abs.y >= incre_abs.z))
    {
        err_1 = incre_err.x - 1;
        err_2 = incre_err.z - 1;
        flag[create_block(map_struct, block_near)] = 2;
        for (int i = 0; i < incre_abs.y; ++i)
        {
            if (err_1 > 0)
            {
                block_near.x += increment.x;
                err_1 -= incre_err.y;
            }

            if (err_2 > 0)
            {
                block_near.z += increment.z;
                err_2 -= incre_err.y;
            }

            err_1 += incre_err.x;
            err_2 += incre_err.z;
            block_near.y += increment.y;
            flag[create_block(map_struct, block_near)] = 2;
        }
    }
    else
    {
        err_1 = incre_err.y - 1;
        err_2 = incre_err.x - 1;
        flag[create_block(map_struct, block_near)] = 2;
        for (int i = 0; i < incre_abs.z; ++i)
        {
            if (err_1 > 0)
            {
                block_near.y += increment.y;
                err_1 -= incre_err.z;
            }

            if (err_2 > 0)
            {
                block_near.x += increment.x;
                err_2 -= incre_err.z;
            }

            err_1 += incre_err.y;
            err_2 += incre_err.x;
            block_near.z += increment.z;
            flag[create_block(map_struct, block_near)] = 2;
        }
    }
}

__global__ void update_map_kernel(MapStorage map_struct,
                                  HashEntry *visible_blocks,
                                  uint count_visible_block,
                                  cv::cuda::PtrStepSz<float> depth,
                                  Matrix3x4f inv_pose,
                                  float fx, float fy,
                                  float cx, float cy)
{
    if (blockIdx.x >= param.num_total_hash_entries_ || blockIdx.x >= count_visible_block)
        return;

    HashEntry &current = visible_blocks[blockIdx.x];

    Vector3i voxel_pos = blockPosToVoxelPos(current.pos_);
    float dist_thresh = param.truncation_dist();
    float inv_dist_thresh = 1.0 / dist_thresh;

#pragma unroll
    for (int block_idx_z = 0; block_idx_z < 8; ++block_idx_z)
    {
        Vector3i local_pos = Vector3i(threadIdx.x, threadIdx.y, block_idx_z);
        Vector3f pt = inv_pose(voxelPosToWorldPt(voxel_pos + local_pos, param.voxel_size));

        int u = __float2int_rd(fx * pt.x / pt.z + cx + 0.5);
        int v = __float2int_rd(fy * pt.y / pt.z + cy + 0.5);
        if (u < 0 || v < 0 || u > depth.cols - 1 || v > depth.rows - 1)
            continue;

        float dist = depth.ptr(v)[u];
        if (isnan(dist) || dist < 1e-2 || dist > param.zmax_update || dist < param.zmin_update)
            continue;

        float sdf = dist - pt.z;
        if (sdf < -dist_thresh)
            continue;

        sdf = fmin(1.0f, sdf * inv_dist_thresh);
        const int local_idx = localPosToLocalIdx(local_pos);
        Voxel &voxel = map_struct.voxels_[current.ptr_ + local_idx];

        auto sdf_p = voxel.getSDF();
        auto weight_p = voxel.getWeight();
        auto weight = 1 / (dist);

        if (weight_p < 1e-3)
        {
            voxel.setSDF(sdf);
            voxel.setWeight(weight);
            continue;
        }

        // fuse depth
        sdf_p = (sdf_p * weight_p + sdf * weight) / (weight_p + weight);
        voxel.setSDF(sdf_p);
        voxel.setWeight(weight_p + weight);
    }
}

__global__ void update_map_with_colour_kernel(MapStorage map_struct,
                                              HashEntry *visible_blocks,
                                              uint count_visible_block,
                                              cv::cuda::PtrStepSz<float> depth,
                                              cv::cuda::PtrStepSz<Vector3c> image,
                                              Matrix3x4f inv_pose,
                                              float fx, float fy,
                                              float cx, float cy)
{
    if (blockIdx.x >= param.num_total_hash_entries_ || blockIdx.x >= count_visible_block)
        return;

    HashEntry &current = visible_blocks[blockIdx.x];

    Vector3i voxel_pos = blockPosToVoxelPos(current.pos_);
    float dist_thresh = param.truncation_dist();
    float inv_dist_thresh = 1.0 / dist_thresh;

#pragma unroll
    for (int block_idx_z = 0; block_idx_z < 8; ++block_idx_z)
    {
        Vector3i local_pos = Vector3i(threadIdx.x, threadIdx.y, block_idx_z);
        Vector3f pt = inv_pose(voxelPosToWorldPt(voxel_pos + local_pos, param.voxel_size));

        int u = __float2int_rd(fx * pt.x / pt.z + cx + 0.5);
        int v = __float2int_rd(fy * pt.y / pt.z + cy + 0.5);
        if (u < 0 || v < 0 || u > depth.cols - 1 || v > depth.rows - 1)
            continue;

        float dist = depth.ptr(v)[u];
        if (isnan(dist) || dist < 1e-2 || dist > param.zmax_update || dist < param.zmin_update)
            continue;

        float sdf = dist - pt.z;
        if (sdf < -dist_thresh)
            continue;

        sdf = fmin(1.0f, sdf * inv_dist_thresh);
        const int local_idx = localPosToLocalIdx(local_pos);
        Voxel &voxel = map_struct.voxels_[current.ptr_ + local_idx];

        auto sdf_p = voxel.getSDF();
        auto weight_p = voxel.getWeight();
        auto weight = 1 / (dist * dist);

        // update colour
        auto colour_new = image.ptr(v)[u];
        auto colour_p = voxel.rgb;

        if (voxel.weight == 0)
        {
            // printf("Initialising map, sdf = %d. \n", sdf);
            voxel.setSDF(sdf);
            voxel.setWeight(weight);
            voxel.rgb = colour_new;
            continue;
        }

        // printf("Fusing depth and color in map. \n");
        // fuse depth
        sdf_p = (sdf_p * weight_p + sdf * weight) / (weight_p + weight);
        voxel.setSDF(sdf_p);
        voxel.setWeight(weight_p + weight);

        // fuse colour
        colour_p = ToVector3c((colour_p * weight_p + colour_new * weight) / (weight_p + weight));
        voxel.rgb = colour_p;
    }
}

__global__ void update_map_weighted_kernel(
    MapStorage map_struct,
    HashEntry *visible_blocks,
    uint count_visible_block,
    cv::cuda::PtrStepSz<float> depth,
    cv::cuda::PtrStepSz<Vector4f> normal,
    cv::cuda::PtrStepSz<Vector3c> image,
    Matrix3x4f inv_pose,
    float fx, float fy,
    float cx, float cy)
{
    if (blockIdx.x >= param.num_total_hash_entries_ || blockIdx.x >= count_visible_block)
        return;

    HashEntry &current = visible_blocks[blockIdx.x];

    if (current.ptr_ < 0)
        return;

    Vector3i voxel_pos = blockPosToVoxelPos(current.pos_);
    float dist_thresh = param.truncation_dist();
    float inv_dist_thresh = 1.0 / dist_thresh;

    #pragma unroll
    for (int block_idx_z = 0; block_idx_z < 8; ++block_idx_z)
    {
        Vector3i local_pos = Vector3i(threadIdx.x, threadIdx.y, block_idx_z);
        Vector3f pt = inv_pose(voxelPosToWorldPt(voxel_pos + local_pos, param.voxel_size));

        int u = __float2int_rd(fx * pt.x / pt.z + cx + 0.5);
        int v = __float2int_rd(fy * pt.y / pt.z + cy + 0.5);
        if (u < 0 || v < 0 || u > depth.cols - 1 || v > depth.rows - 1)
            continue;

        float dist = depth.ptr(v)[u];
        auto n_c = ToVector3(normal.ptr(v)[u]);
        if (isnan(dist) || isnan(n_c.x) || dist > param.zmax_update || dist < param.zmin_update)
            continue;

        float sdf = dist - pt.z;
        if (sdf < -dist_thresh)
            continue;

        sdf = fmin(1.0f, sdf * inv_dist_thresh);
        const int local_idx = localPosToLocalIdx(local_pos);
        Voxel &voxel = map_struct.voxels_[current.ptr_ + local_idx];

        auto sdf_p = voxel.getSDF();
        auto weight_p = voxel.getWeight();
        auto weight = abs(sin(n_c.z)) / (dist * dist);

        // update colour
        auto colour_new = image.ptr(v)[u];
        auto colour_p = voxel.rgb;

        if (voxel.weight == 0)
        {
            voxel.setSDF(sdf);
            voxel.setWeight(weight);
            voxel.rgb = colour_new;
            continue;
        }

        // fuse depth
        sdf_p = (sdf_p * weight_p + sdf * weight) / (weight_p + weight);
        voxel.setSDF(sdf_p);
        voxel.setWeight(weight_p + weight);

        // fuse colour
        if(voxel.label == 0){
            colour_p = ToVector3c((colour_p * weight_p + colour_new * weight) / (weight_p + weight));
            voxel.rgb = colour_p;
        }
    }
}

__global__ void update_map_with_object_kernel(
    MapStorage map_struct,
    HashEntry *visible_blocks,
    uint count_visible_block,
    cv::cuda::PtrStepSz<float> depth,
    cv::cuda::PtrStepSz<Vector3c> image,
    cv::cuda::PtrStepSz<unsigned char> mask,
    Matrix3x4f inv_pose,
    float fx, float fy,
    float cx, float cy)
{
    // current block is a valid entry & inside the visible block
    if (blockIdx.x >= param.num_total_hash_entries_ || blockIdx.x >= count_visible_block)
        return;

    // get current block
    HashEntry &current = visible_blocks[blockIdx.x];

    // check if current block is assigned
    if (current.ptr_ < 0)
        return;

    // get voxel start position and thresh info 
    Vector3i voxel_pos = blockPosToVoxelPos(current.pos_);
    float dist_thresh = param.truncation_dist();

    // loop through all voxels (8, only along z-axis)
    int confident_thre = 2;
    #pragma unroll
    for (int block_idx_z = 0; block_idx_z < 8; ++block_idx_z)
    {
        Vector3i local_pos = Vector3i(threadIdx.x, threadIdx.y, block_idx_z);
        Vector3f pt = inv_pose(voxelPosToWorldPt(voxel_pos + local_pos, param.voxel_size));

        // get pixel coordinates
        int u = __float2int_rd(fx * pt.x / pt.z + cx + 0.5);
        int v = __float2int_rd(fy * pt.y / pt.z + cy + 0.5);
        // check if it falls in the image range
        if (u < 0 || v < 0 || u > depth.cols - 1 || v > depth.rows - 1)
            continue;

        // get object label
        auto label = mask.ptr(v)[u];
        if(label == 0)
            continue;

        // get depth value
        float dist = depth.ptr(v)[u];
        // check if it is in range
        if (isnan(dist) || dist > param.zmax_update || dist < param.zmin_update)
            continue;

        // check if sdf in the truncated range
        float sdf = dist - pt.z;
        if (sdf < -dist_thresh)
            continue;

        const int local_idx = localPosToLocalIdx(local_pos);
        Voxel &voxel = map_struct.voxels_[current.ptr_ + local_idx];

        // DO NOT create new voxels here !!
        auto weight_p = voxel.getWeight();;
        if(weight_p == 0)
            continue;

        // UPDATE object info
        // Vector3c color_new(int(33554431 * int(label) % 255), 
        //                    int(32767 * int(label) % 255),
        //                    int(2097151 * int(label) % 255));
        // the label is consistent
        if(label == voxel.label)
        {
            voxel.count = min(255, voxel.count+1);
        }
        // the label is Inconsistent
        else
        {
            // check with backup label
            if(label == voxel.label_backup)
            {
                voxel.count_backup = min(255, voxel.count_backup+1);
                // voxel.rgb_backup = color_new;
            }
            // update back_up label if the previous one is detected few times only
            // discard new detection if previous one is already very confident
            else
            {
                if(voxel.count_backup <= confident_thre)
                {
                    voxel.label_backup = label;
                    voxel.count_backup = 1;
                    // voxel.rgb_backup = color_new;
                }
            }

            // compare count
            // switch if 1) prev is bg, object detected is confident enough
            //           2) prev is ob, has more count
            if ( (voxel.count < voxel.count_backup && voxel.label !=0) ||
                 (voxel.count_backup > confident_thre && voxel.label ==0) )
            {
                short tmp = voxel.label;
                voxel.label = voxel.label_backup;
                voxel.label_backup = tmp;

                tmp = voxel.count;
                voxel.count = voxel.count_backup;
                voxel.count_backup = tmp;

                // Vector3c tmpC = voxel.rgb;
                // voxel.rgb = voxel.rgb_backup;
                // voxel.rgb_backup = tmpC;
            }
        }
    } // end block_idx_z
}

void update(
    MapStorage map_struct,
    MapState state,
    const cv::cuda::GpuMat depth,
    const cv::cuda::GpuMat image,
    const Sophus::SE3d &frame_pose,
    const Eigen::Matrix3f K,
    cv::cuda::GpuMat &cv_flag,
    cv::cuda::GpuMat &cv_pos_array,
    HashEntry *visible_blocks,
    uint &visible_block_count)
{
    if (cv_flag.empty())
        cv_flag.create(1, state.num_total_hash_entries_, CV_8UC1);
    if (cv_pos_array.empty())
        cv_pos_array.create(1, state.num_total_hash_entries_, CV_32SC1);

    thrust::device_ptr<uchar> flag(cv_flag.ptr<uchar>());
    thrust::device_ptr<int> pos_array(cv_pos_array.ptr<int>());

    const int cols = depth.cols;
    const int rows = depth.rows;

    dim3 thread(8, 8);
    dim3 block(div_up(cols, thread.x), div_up(rows, thread.y));

    create_blocks_kernel<<<block, thread>>>(
        map_struct,
        depth,
        1.0/K(0,0),
        1.0/K(1,1),
        K(0,2), K(1,2),
        frame_pose.cast<float>().matrix3x4(),
        flag.get());

    thread = dim3(MAX_THREAD);
    block = dim3(div_up(state.num_total_hash_entries_, thread.x));

    check_visibility_flag_kernel<<<block, thread>>>(
        map_struct,
        flag.get(),
        frame_pose.inverse().cast<float>().matrix3x4(),
        cols, rows,
        K(0,0), K(1,1),
        K(0,2), K(1,2));

    thrust::exclusive_scan(flag, flag + state.num_total_hash_entries_, pos_array);

    copy_visible_block_kernel<<<block, thread>>>(
        map_struct.hash_table_,
        visible_blocks,
        flag.get(),
        pos_array.get());

    visible_block_count = pos_array[state.num_total_hash_entries_ - 1];
    
    printf("visible_block_count = %d\n", visible_block_count);
    if (visible_block_count == 0)
        return;

    thread = dim3(8, 8);
    block = dim3(visible_block_count);

    update_map_with_colour_kernel<<<block, thread>>>(
        map_struct,
        visible_blocks,
        visible_block_count,
        depth, image,
        frame_pose.inverse().cast<float>().matrix3x4(),
        K(0,0), K(1,1),
        K(0,2), K(1,2));
}

// MAIN FUSING FUNCTION
void update_weighted(
    MapStorage map_struct,
    MapState state,
    const cv::cuda::GpuMat depth,
    const cv::cuda::GpuMat normal,
    const cv::cuda::GpuMat image,
    const Sophus::SE3d &frame_pose,
    const Eigen::Matrix3f K,
    cv::cuda::GpuMat &cv_flag,
    cv::cuda::GpuMat &cv_pos_array,
    HashEntry *visible_blocks,
    uint &visible_block_count)
{
    if (cv_flag.empty())
        cv_flag.create(1, state.num_total_hash_entries_, CV_8UC1);
    if (cv_pos_array.empty())
        cv_pos_array.create(1, state.num_total_hash_entries_, CV_32SC1);

    thrust::device_ptr<uchar> flag(cv_flag.ptr<uchar>());
    thrust::device_ptr<int> pos_array(cv_pos_array.ptr<int>());

    const int cols = depth.cols;
    const int rows = depth.rows;

    dim3 thread(8, 8);
    dim3 block(div_up(cols, thread.x), div_up(rows, thread.y));

    create_blocks_kernel<<<block, thread>>>(
        map_struct,
        depth,
        1.0/K(0,0),
        1.0/K(1,1),
        K(0,2), K(1,2),
        frame_pose.cast<float>().matrix3x4(),
        flag.get());
    cudaDeviceSynchronize();
    safe_call(cudaGetLastError());

    thread = dim3(MAX_THREAD);
    block = dim3(div_up(state.num_total_hash_entries_, thread.x));

    check_visibility_flag_kernel<<<block, thread>>>(
        map_struct,
        flag.get(),
        frame_pose.inverse().cast<float>().matrix3x4(),
        cols, rows,
        K(0,0), K(1,1),
        K(0,2), K(1,2));
    cudaDeviceSynchronize();
    safe_call(cudaGetLastError());

    thrust::exclusive_scan(flag, flag + state.num_total_hash_entries_, pos_array);

    copy_visible_block_kernel<<<block, thread>>>(
        map_struct.hash_table_,
        visible_blocks,
        flag.get(),
        pos_array.get());
    cudaDeviceSynchronize();
    safe_call(cudaGetLastError());

    visible_block_count = pos_array[state.num_total_hash_entries_ - 1];

    if (visible_block_count == 0)
        return;

    thread = dim3(8, 8);
    block = dim3(visible_block_count);

    update_map_weighted_kernel<<<block, thread>>>(
        map_struct,
        visible_blocks,
        visible_block_count,
        depth,
        normal,
        image,
        frame_pose.inverse().cast<float>().matrix3x4(),
        K(0,0), K(1,1),
        K(0,2), K(1,2));
    cudaDeviceSynchronize();
    safe_call(cudaGetLastError());
}

void create_new_block(
    MapStorage map_struct,
    MapState state,
    const cv::cuda::GpuMat depth,
    const Sophus::SE3d &frame_pose,
    const Eigen::Matrix3f K,
    cv::cuda::GpuMat &cv_flag,
    cv::cuda::GpuMat &cv_pos_array,
    HashEntry *visible_blocks,
    uint &visible_block_count)
{
    if (cv_flag.empty())
        cv_flag.create(1, state.num_total_hash_entries_, CV_8UC1);
    if (cv_pos_array.empty())
        cv_pos_array.create(1, state.num_total_hash_entries_, CV_32SC1);

    thrust::device_ptr<uchar> flag(cv_flag.ptr<uchar>());
    thrust::device_ptr<int> pos_array(cv_pos_array.ptr<int>());

    const int cols = depth.cols;
    const int rows = depth.rows;

    dim3 thread(8, 8);
    dim3 block(div_up(cols, thread.x), div_up(rows, thread.y));

    create_blocks_kernel<<<block, thread>>>(
        map_struct,
        depth,
        1.0/K(0,0),
        1.0/K(1,1),
        K(0,2), K(1,2),
        frame_pose.cast<float>().matrix3x4(),
        flag.get());
    cudaDeviceSynchronize();
    safe_call(cudaGetLastError());
    printf("created blocks   ");

    thread = dim3(MAX_THREAD);
    block = dim3(div_up(state.num_total_hash_entries_, thread.x));

    check_visibility_flag_kernel<<<block, thread>>>(
        map_struct,
        flag.get(),
        frame_pose.inverse().cast<float>().matrix3x4(),
        cols, rows,
        K(0,0), K(1,1),
        K(0,2), K(1,2));
    cudaDeviceSynchronize();
    safe_call(cudaGetLastError());
    printf("checked visibility flag   ");

    thrust::exclusive_scan(flag, flag + state.num_total_hash_entries_, pos_array);

    copy_visible_block_kernel<<<block, thread>>>(
        map_struct.hash_table_,
        visible_blocks,
        flag.get(),
        pos_array.get());
    cudaDeviceSynchronize();
    safe_call(cudaGetLastError());
    printf("copied visible blocks   ");

    visible_block_count = pos_array[state.num_total_hash_entries_ - 1];

    if (visible_block_count == 0)
        return;

}

void check_visibility(
    MapStorage map_struct,
    MapState state,
    const cv::cuda::GpuMat depth,
    const Sophus::SE3d &frame_pose,
    const Eigen::Matrix3f K,
    cv::cuda::GpuMat &cv_flag,
    cv::cuda::GpuMat &cv_pos_array,
    HashEntry *visible_blocks,
    uint &visible_block_count)
{
    if (cv_flag.empty())
        cv_flag.create(1, state.num_total_hash_entries_, CV_8UC1);
    if (cv_pos_array.empty())
        cv_pos_array.create(1, state.num_total_hash_entries_, CV_32SC1);

    thrust::device_ptr<uchar> flag(cv_flag.ptr<uchar>());
    thrust::device_ptr<int> pos_array(cv_pos_array.ptr<int>());

    const int cols = depth.cols;
    const int rows = depth.rows;

    dim3 thread(MAX_THREAD);
    dim3 block(div_up(state.num_total_hash_entries_, thread.x));

    check_visibility_flag_kernel<<<block, thread>>>(
        map_struct,
        flag.get(),
        frame_pose.inverse().cast<float>().matrix3x4(),
        cols, rows,
        K(0,0), K(1,1),
        K(0,2), K(1,2));
    cudaDeviceSynchronize();
    safe_call(cudaGetLastError());

    thrust::exclusive_scan(flag, flag + state.num_total_hash_entries_, pos_array);

    copy_visible_block_kernel<<<block, thread>>>(
        map_struct.hash_table_,
        visible_blocks,
        flag.get(),
        pos_array.get());
    cudaDeviceSynchronize();
    safe_call(cudaGetLastError());

    visible_block_count = pos_array[state.num_total_hash_entries_ - 1];
}

void color_objects(
    MapStorage map_struct,
    MapState state,
    const cv::cuda::GpuMat depth,
    const cv::cuda::GpuMat image,
    const cv::cuda::GpuMat mask,
    const Sophus::SE3d &frame_pose,
    const Eigen::Matrix3f K,
    cv::cuda::GpuMat &cv_flag,
    cv::cuda::GpuMat &cv_pos_array,
    HashEntry *visible_blocks,
    uint &visible_block_count)
{
    if (cv_flag.empty())
        cv_flag.create(1, state.num_total_hash_entries_, CV_8UC1);
    if (cv_pos_array.empty())
        cv_pos_array.create(1, state.num_total_hash_entries_, CV_32SC1);

    thrust::device_ptr<uchar> flag(cv_flag.ptr<uchar>());
    thrust::device_ptr<int> pos_array(cv_pos_array.ptr<int>());

    const int cols = depth.cols;
    const int rows = depth.rows;

    dim3 thread, block;

    // find all visible blocks
    thread = dim3(MAX_THREAD);
    block = dim3(div_up(state.num_total_hash_entries_, thread.x));
    check_visibility_flag_kernel<<<block, thread>>>(
        map_struct,
        flag.get(),
        frame_pose.inverse().cast<float>().matrix3x4(),
        cols, rows,
        K(0,0), K(1,1),
        K(0,2), K(1,2));

    thrust::exclusive_scan(flag, flag + state.num_total_hash_entries_, pos_array);

    copy_visible_block_kernel<<<block, thread>>>(
        map_struct.hash_table_,
        visible_blocks,
        flag.get(),
        pos_array.get());

    visible_block_count = pos_array[state.num_total_hash_entries_ - 1];

    if (visible_block_count == 0)
        return;

    // update object information in the map
    thread = dim3(8, 8);
    block = dim3(visible_block_count);
    update_map_with_object_kernel<<<block, thread>>>(
        map_struct,
        visible_blocks,
        visible_block_count,
        depth,
        image,
        mask,
        frame_pose.inverse().cast<float>().matrix3x4(),
        K(0,0), K(1,1),
        K(0,2), K(1,2));
}

// __global__ void update_cuboids_dimension_kernel(
//     cv::cuda::PtrStep<Vector4f> vmap,
//     cv::cuda::PtrStepSz<unsigned char> mask,
//     cv::cuda::PtrStepSz<float> bbox,
//     unsigned char label,
//     cv::cuda::PtrStepSz<float> &cuboid)
// {
//     float x_min, y_min, x_max, y_max;   // x -> cols, y -> rows
//     x_min = bbox.ptr(0)[0];
//     y_min = bbox.ptr(0)[1];
//     x_max = bbox.ptr(0)[2];
//     y_max = bbox.ptr(0)[3];
//     int v = threadIdx.x;
//     int u = blockIdx.x;

//     // we only care the region inside the bbox
//     if(u<x_min || u>x_max || v<y_min || v>y_max)
//         return;

//     // check label consistency
//     if(mask.ptr(v)[u] != label)
//         return;

//     Vector4f vertex = vmap.ptr(v)[u];
//     float tmp_x = vertex.x / vertex.w;
//     float tmp_y = vertex.y / vertex.w;
//     float tmp_z = vertex.z / vertex.w;


// }

// void estimate_cuboids(
//     const cv::cuda::GpuMat vmap,
//     const cv::cuda::GpuMat mask,
//     const cv::cuda::GpuMat bbox,
//     const unsigned char label,
//     cv::cuda::GpuMat &cuboid)
// {
//     const int cols = vmap.cols;
//     const int rows = vmap.rows;

//     cuboid.create(1, 6. CV_32F);

//     dim3 thread, block;
//     thread = dim3(rows);
//     block = dim3(cols);
//     estimate_cuboids<<<block, thread>>>(
//         vmap,
//         mask,
//         bbox,
//         label,
//         cuboid);

// }


} // namespace cuda
} // namespace fusion