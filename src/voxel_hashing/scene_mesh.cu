#include "voxel_hashing/map_proc.h"
#include "math/matrix_type.h"
#include "math/vector_type.h"
#include "utils/safe_call.h"
#include "voxel_hashing/prefix_sum.h"
#include "voxel_hashing/voxel_hashing.h"
#include "voxel_hashing/device_tables.h"

namespace fusion
{
namespace cuda
{

struct BuildVertexArray
{
    MapStorage map_struct;

    Vector3f *triangles;
    HashEntry *block_array;
    uint *block_count;
    uint *triangle_count;
    Vector3f *surface_normal;

    FUSION_DEVICE inline void select_blocks() const
    {
        int x = blockDim.x * blockIdx.x + threadIdx.x;

        __shared__ bool scan_required;

        if (x == 0)
            scan_required = false;

        __syncthreads();

        uint val = 0;
        if (x < param.num_total_hash_entries_ && map_struct.hash_table_[x].ptr_ >= 0)
        {
            scan_required = true;
            val = 1;
        }

        __syncthreads();

        if (scan_required)
        {
            int offset = exclusive_scan<1024>(val, block_count);
            if (offset != -1)
            {
                block_array[offset] = map_struct.hash_table_[x];
            }
        }
    }

    FUSION_DEVICE inline float read_sdf(Vector3f pt, bool &valid) const
    {
        Voxel *voxel = NULL;
        findVoxel(map_struct, ToVector3i(pt), voxel);
        if (voxel && voxel->weight != 0)
        {
            valid = true;
            return voxel->getSDF();
        }
        else
        {
            valid = false;
            return 0;
        }
    }

    FUSION_DEVICE inline bool read_sdf_list(float *sdf, Vector3i pos) const
    {
        bool valid = false;
        sdf[0] = read_sdf(pos + Vector3f(0, 0, 0), valid);
        if (!valid)
            return false;

        sdf[1] = read_sdf(pos + Vector3f(1, 0, 0), valid);
        if (!valid)
            return false;

        sdf[2] = read_sdf(pos + Vector3f(1, 1, 0), valid);
        if (!valid)
            return false;

        sdf[3] = read_sdf(pos + Vector3f(0, 1, 0), valid);
        if (!valid)
            return false;

        sdf[4] = read_sdf(pos + Vector3f(0, 0, 1), valid);
        if (!valid)
            return false;

        sdf[5] = read_sdf(pos + Vector3f(1, 0, 1), valid);
        if (!valid)
            return false;

        sdf[6] = read_sdf(pos + Vector3f(1, 1, 1), valid);
        if (!valid)
            return false;

        sdf[7] = read_sdf(pos + Vector3f(0, 1, 1), valid);
        if (!valid)
            return false;

        return true;
    }

    FUSION_DEVICE inline float interpolate_sdf(float &v1, float &v2) const
    {
        if (fabs(0 - v1) < 1e-6)
            return 0;
        if (fabs(0 - v2) < 1e-6)
            return 1;
        if (fabs(v1 - v2) < 1e-6)
            return 0;
        return (0 - v1) / (v2 - v1);
    }

    FUSION_DEVICE inline int make_vertex(Vector3f *vertex_array, const Vector3i pos)
    {
        float sdf[8];

        if (!read_sdf_list(sdf, pos))
            return -1;

        int cube_index = 0;
        if (sdf[0] < 0)
            cube_index |= 1;
        if (sdf[1] < 0)
            cube_index |= 2;
        if (sdf[2] < 0)
            cube_index |= 4;
        if (sdf[3] < 0)
            cube_index |= 8;
        if (sdf[4] < 0)
            cube_index |= 16;
        if (sdf[5] < 0)
            cube_index |= 32;
        if (sdf[6] < 0)
            cube_index |= 64;
        if (sdf[7] < 0)
            cube_index |= 128;

        if (edge_table[cube_index] == 0)
            return -1;

        if (edge_table[cube_index] & 1)
        {
            float val = interpolate_sdf(sdf[0], sdf[1]);
            vertex_array[0] = pos + Vector3f(val, 0, 0);
        }
        if (edge_table[cube_index] & 2)
        {
            float val = interpolate_sdf(sdf[1], sdf[2]);
            vertex_array[1] = pos + Vector3f(1, val, 0);
        }
        if (edge_table[cube_index] & 4)
        {
            float val = interpolate_sdf(sdf[2], sdf[3]);
            vertex_array[2] = pos + Vector3f(1 - val, 1, 0);
        }
        if (edge_table[cube_index] & 8)
        {
            float val = interpolate_sdf(sdf[3], sdf[0]);
            vertex_array[3] = pos + Vector3f(0, 1 - val, 0);
        }
        if (edge_table[cube_index] & 16)
        {
            float val = interpolate_sdf(sdf[4], sdf[5]);
            vertex_array[4] = pos + Vector3f(val, 0, 1);
        }
        if (edge_table[cube_index] & 32)
        {
            float val = interpolate_sdf(sdf[5], sdf[6]);
            vertex_array[5] = pos + Vector3f(1, val, 1);
        }
        if (edge_table[cube_index] & 64)
        {
            float val = interpolate_sdf(sdf[6], sdf[7]);
            vertex_array[6] = pos + Vector3f(1 - val, 1, 1);
        }
        if (edge_table[cube_index] & 128)
        {
            float val = interpolate_sdf(sdf[7], sdf[4]);
            vertex_array[7] = pos + Vector3f(0, 1 - val, 1);
        }
        if (edge_table[cube_index] & 256)
        {
            float val = interpolate_sdf(sdf[0], sdf[4]);
            vertex_array[8] = pos + Vector3f(0, 0, val);
        }
        if (edge_table[cube_index] & 512)
        {
            float val = interpolate_sdf(sdf[1], sdf[5]);
            vertex_array[9] = pos + Vector3f(1, 0, val);
        }
        if (edge_table[cube_index] & 1024)
        {
            float val = interpolate_sdf(sdf[2], sdf[6]);
            vertex_array[10] = pos + Vector3f(1, 1, val);
        }
        if (edge_table[cube_index] & 2048)
        {
            float val = interpolate_sdf(sdf[3], sdf[7]);
            vertex_array[11] = pos + Vector3f(0, 1, val);
        }

        return cube_index;
    }

    template <bool compute_normal = false>
    FUSION_DEVICE inline void operator()()
    {
        int x = blockIdx.y * gridDim.x + blockIdx.x;
        if (*triangle_count >= param.num_max_mesh_triangles_ || x >= *block_count)
            return;

        Vector3f vertex_array[12];
        Vector3i pos = block_array[x].pos_ * BLOCK_SIZE;
        auto factor = param.voxel_size;

        for (int voxel_id = 0; voxel_id < BLOCK_SIZE; ++voxel_id)
        {
            Vector3i local_pos = Vector3i(threadIdx.x, threadIdx.y, voxel_id);
            int cube_index = make_vertex(vertex_array, pos + local_pos);
            if (cube_index <= 0)
                continue;

            for (int i = 0; triangle_table[cube_index][i] != -1; i += 3)
            {
                uint triangleId = atomicAdd(triangle_count, 1);

                if (triangleId < param.num_max_mesh_triangles_)
                {
                    triangles[triangleId * 3] = vertex_array[triangle_table[cube_index][i]] * factor;
                    triangles[triangleId * 3 + 1] = vertex_array[triangle_table[cube_index][i + 1]] * factor;
                    triangles[triangleId * 3 + 2] = vertex_array[triangle_table[cube_index][i + 2]] * factor;

                    if (compute_normal)
                    {
                        surface_normal[triangleId * 3] = normalised((triangles[triangleId * 3 + 1] - triangles[triangleId * 3]).cross(triangles[triangleId * 3 + 2] - triangles[triangleId * 3]));
                        surface_normal[triangleId * 3 + 1] = surface_normal[triangleId * 3 + 2] = surface_normal[triangleId * 3];
                    }
                }
            }
        }
    }
};

__global__ void select_blocks_kernel(BuildVertexArray bva)
{
    bva.select_blocks();
}

__global__ void generate_vertex_array_kernel(BuildVertexArray bva)
{
    bva.operator()<false>();
}

void create_mesh_vertex_only(
    MapStorage map_struct,
    MapState state,
    uint &block_count,
    HashEntry *block_list,
    uint &triangle_count,
    void *vertex_data)
{
    uint *cuda_block_count;
    uint *cuda_triangle_count;
    safe_call(cudaMalloc(&cuda_block_count, sizeof(uint)));
    safe_call(cudaMalloc(&cuda_triangle_count, sizeof(uint)));
    safe_call(cudaMemset(cuda_block_count, 0, sizeof(uint)));
    safe_call(cudaMemset(cuda_triangle_count, 0, sizeof(uint)));

    BuildVertexArray bva;
    bva.map_struct = map_struct;
    bva.block_array = block_list;
    bva.block_count = cuda_block_count;
    bva.triangle_count = cuda_triangle_count;
    bva.triangles = static_cast<Vector3f *>(vertex_data);

    dim3 thread(1024);
    dim3 block = dim3(div_up(state.num_total_hash_entries_, thread.x));

    select_blocks_kernel<<<block, thread>>>(bva);

    safe_call(cudaMemcpy(&block_count, cuda_block_count, sizeof(uint), cudaMemcpyDeviceToHost));
    if (block_count == 0)
        return;

    thread = dim3(8, 8);
    block = dim3(div_up(block_count, 16), 16);

    generate_vertex_array_kernel<<<block, thread>>>(bva);

    safe_call(cudaMemcpy(&triangle_count, cuda_triangle_count, sizeof(uint), cudaMemcpyDeviceToHost));
    triangle_count = std::min(triangle_count, (uint)state.num_max_mesh_triangles_);

    safe_call(cudaFree(cuda_block_count));
    safe_call(cudaFree(cuda_triangle_count));
}

__global__ void generate_vertex_and_normal_array_kernel(BuildVertexArray bva)
{
    bva.operator()<true>();
}

void create_mesh_with_normal(
    MapStorage map_struct,
    MapState state,
    uint &block_count,
    HashEntry *block_list,
    uint &triangle_count,
    void *vertex_data,
    void *vertex_normal)
{
    uint *cuda_block_count;
    uint *cuda_triangle_count;
    safe_call(cudaMalloc(&cuda_block_count, sizeof(uint)));
    safe_call(cudaMalloc(&cuda_triangle_count, sizeof(uint)));
    safe_call(cudaMemset(cuda_block_count, 0, sizeof(uint)));
    safe_call(cudaMemset(cuda_triangle_count, 0, sizeof(uint)));

    BuildVertexArray bva;
    bva.map_struct = map_struct;
    bva.block_array = block_list;
    bva.block_count = cuda_block_count;
    bva.triangle_count = cuda_triangle_count;
    bva.triangles = static_cast<Vector3f *>(vertex_data);
    bva.surface_normal = static_cast<Vector3f *>(vertex_normal);

    dim3 thread(1024);
    dim3 block = dim3(div_up(state.num_total_hash_entries_, thread.x));

    select_blocks_kernel<<<block, thread>>>(bva);

    safe_call(cudaMemcpy(&block_count, cuda_block_count, sizeof(uint), cudaMemcpyDeviceToHost));
    if (block_count == 0)
        return;

    thread = dim3(8, 8);
    block = dim3(div_up(block_count, 16), 16);

    generate_vertex_and_normal_array_kernel<<<block, thread>>>(bva);

    safe_call(cudaMemcpy(&triangle_count, cuda_triangle_count, sizeof(uint), cudaMemcpyDeviceToHost));
    triangle_count = std::min(triangle_count, (uint)state.num_max_mesh_triangles_);

    safe_call(cudaFree(cuda_block_count));
    safe_call(cudaFree(cuda_triangle_count));
}

struct BuildVertexAndColourArray
{
    MapStorage map_struct;

    Vector3f *triangles;
    HashEntry *block_array;
    uint *block_count;
    uint *triangle_count;
    Vector3c *vertex_colour;

    FUSION_DEVICE inline void select_blocks() const
    {
        int x = blockDim.x * blockIdx.x + threadIdx.x;

        __shared__ bool scan_required;

        if (x == 0)
            scan_required = false;

        __syncthreads();

        uint val = 0;
        if (x < param.num_total_hash_entries_ && map_struct.hash_table_[x].ptr_ >= 0)
        {
            scan_required = true;
            val = 1;
        }

        __syncthreads();

        if (scan_required)
        {
            int offset = exclusive_scan<1024>(val, block_count);
            if (offset != -1)
            {
                block_array[offset] = map_struct.hash_table_[x];
            }
        }
    }

    FUSION_DEVICE inline void read_sdf_and_colour(Vector3f pt, bool &valid, float &sdf, Vector3c &colour) const
    {
        Voxel *vx = NULL;
        findVoxel(map_struct, ToVector3i(pt), vx);
        if (vx && vx->getWeight() > 1e-3)
        {
            valid = true;
            sdf = vx->getSDF();
            colour = vx->rgb;
        }
        else
        {
            valid = false;
        }
    }

    FUSION_DEVICE inline bool read_sdf_and_colour_list(float *sdf, Vector3c *colour, Vector3i pos) const
    {
        bool valid = false;
        read_sdf_and_colour(pos + Vector3f(0, 0, 0), valid, sdf[0], colour[0]);
        if (!valid)
            return false;

        read_sdf_and_colour(pos + Vector3f(1, 0, 0), valid, sdf[1], colour[1]);
        if (!valid)
            return false;

        read_sdf_and_colour(pos + Vector3f(1, 1, 0), valid, sdf[2], colour[2]);
        if (!valid)
            return false;

        read_sdf_and_colour(pos + Vector3f(0, 1, 0), valid, sdf[3], colour[3]);
        if (!valid)
            return false;

        read_sdf_and_colour(pos + Vector3f(0, 0, 1), valid, sdf[4], colour[4]);
        if (!valid)
            return false;

        read_sdf_and_colour(pos + Vector3f(1, 0, 1), valid, sdf[5], colour[5]);
        if (!valid)
            return false;

        read_sdf_and_colour(pos + Vector3f(1, 1, 1), valid, sdf[6], colour[6]);
        if (!valid)
            return false;

        read_sdf_and_colour(pos + Vector3f(0, 1, 1), valid, sdf[7], colour[7]);
        if (!valid)
            return false;

        return true;
    }

    FUSION_DEVICE inline float interpolate_sdf(float &v1, float &v2) const
    {
        if (fabs(0 - v1) < 1e-6)
            return 0;
        if (fabs(0 - v2) < 1e-6)
            return 1;
        if (fabs(v1 - v2) < 1e-6)
            return 0;
        return (0 - v1) / (v2 - v1);
    }

    FUSION_DEVICE inline int make_vertex_and_colour(Vector3f *vertex_array, Vector3c *colour_array, const Vector3i pos)
    {
        float sdf[8];

        if (!read_sdf_and_colour_list(sdf, colour_array, pos))
            return -1;

        int cube_index = 0;
        if (sdf[0] < 0)
            cube_index |= 1;
        if (sdf[1] < 0)
            cube_index |= 2;
        if (sdf[2] < 0)
            cube_index |= 4;
        if (sdf[3] < 0)
            cube_index |= 8;
        if (sdf[4] < 0)
            cube_index |= 16;
        if (sdf[5] < 0)
            cube_index |= 32;
        if (sdf[6] < 0)
            cube_index |= 64;
        if (sdf[7] < 0)
            cube_index |= 128;

        if (edge_table[cube_index] == 0)
            return -1;

        if (edge_table[cube_index] & 1)
        {
            float val = interpolate_sdf(sdf[0], sdf[1]);
            vertex_array[0] = pos + Vector3f(val, 0, 0);
        }
        if (edge_table[cube_index] & 2)
        {
            float val = interpolate_sdf(sdf[1], sdf[2]);
            vertex_array[1] = pos + Vector3f(1, val, 0);
        }
        if (edge_table[cube_index] & 4)
        {
            float val = interpolate_sdf(sdf[2], sdf[3]);
            vertex_array[2] = pos + Vector3f(1 - val, 1, 0);
        }
        if (edge_table[cube_index] & 8)
        {
            float val = interpolate_sdf(sdf[3], sdf[0]);
            vertex_array[3] = pos + Vector3f(0, 1 - val, 0);
        }
        if (edge_table[cube_index] & 16)
        {
            float val = interpolate_sdf(sdf[4], sdf[5]);
            vertex_array[4] = pos + Vector3f(val, 0, 1);
        }
        if (edge_table[cube_index] & 32)
        {
            float val = interpolate_sdf(sdf[5], sdf[6]);
            vertex_array[5] = pos + Vector3f(1, val, 1);
        }
        if (edge_table[cube_index] & 64)
        {
            float val = interpolate_sdf(sdf[6], sdf[7]);
            vertex_array[6] = pos + Vector3f(1 - val, 1, 1);
        }
        if (edge_table[cube_index] & 128)
        {
            float val = interpolate_sdf(sdf[7], sdf[4]);
            vertex_array[7] = pos + Vector3f(0, 1 - val, 1);
        }
        if (edge_table[cube_index] & 256)
        {
            float val = interpolate_sdf(sdf[0], sdf[4]);
            vertex_array[8] = pos + Vector3f(0, 0, val);
            colour_array[8] = colour_array[0];
        }
        if (edge_table[cube_index] & 512)
        {
            float val = interpolate_sdf(sdf[1], sdf[5]);
            vertex_array[9] = pos + Vector3f(1, 0, val);
            colour_array[9] = colour_array[1];
        }
        if (edge_table[cube_index] & 1024)
        {
            float val = interpolate_sdf(sdf[2], sdf[6]);
            vertex_array[10] = pos + Vector3f(1, 1, val);
            colour_array[10] = colour_array[2];
        }
        if (edge_table[cube_index] & 2048)
        {
            float val = interpolate_sdf(sdf[3], sdf[7]);
            vertex_array[11] = pos + Vector3f(0, 1, val);
            colour_array[11] = colour_array[3];
        }

        return cube_index;
    }

    FUSION_DEVICE inline void operator()()
    {
        int x = blockIdx.y * gridDim.x + blockIdx.x;
        if (*triangle_count >= param.num_max_mesh_triangles_ || x >= *block_count)
            return;

        Vector3f vertex_array[12];
        Vector3c colour_array[12];
        Vector3i pos = block_array[x].pos_ * BLOCK_SIZE;
        auto factor = param.voxel_size;

        for (int voxel_id = 0; voxel_id < BLOCK_SIZE; ++voxel_id)
        {
            Vector3i local_pos = Vector3i(threadIdx.x, threadIdx.y, voxel_id);
            int cube_index = make_vertex_and_colour(vertex_array, colour_array, pos + local_pos);
            if (cube_index <= 0)
                continue;

            for (int i = 0; triangle_table[cube_index][i] != -1; i += 3)
            {
                uint triangleId = atomicAdd(triangle_count, 1);

                if (triangleId < param.num_max_mesh_triangles_)
                {
                    triangles[triangleId * 3] = vertex_array[triangle_table[cube_index][i]] * factor;
                    triangles[triangleId * 3 + 1] = vertex_array[triangle_table[cube_index][i + 1]] * factor;
                    triangles[triangleId * 3 + 2] = vertex_array[triangle_table[cube_index][i + 2]] * factor;
                    vertex_colour[triangleId * 3] = colour_array[triangle_table[cube_index][i]];
                    vertex_colour[triangleId * 3 + 1] = colour_array[triangle_table[cube_index][i + 1]];
                    vertex_colour[triangleId * 3 + 2] = colour_array[triangle_table[cube_index][i + 2]];
                }
            }
        }
    }
};

__global__ void select_blocks_coloured_kernel(BuildVertexAndColourArray delegate)
{
    delegate.select_blocks();
}

__global__ void generate_vertex_and_colour_array_kernel(BuildVertexAndColourArray delegate)
{
    delegate();
}

void create_mesh_with_colour(
    MapStorage map_struct,
    MapState state,
    uint &block_count,
    HashEntry *block_list,
    uint &triangle_count,
    void *vertex_data,
    void *vertex_colour)
{
    uint *cuda_block_count;
    uint *cuda_triangle_count;
    safe_call(cudaMalloc(&cuda_block_count, sizeof(uint)));
    safe_call(cudaMalloc(&cuda_triangle_count, sizeof(uint)));
    safe_call(cudaMemset(cuda_block_count, 0, sizeof(uint)));
    safe_call(cudaMemset(cuda_triangle_count, 0, sizeof(uint)));

    BuildVertexAndColourArray delegate;
    delegate.map_struct = map_struct;
    delegate.block_array = block_list;
    delegate.block_count = cuda_block_count;
    delegate.triangle_count = cuda_triangle_count;
    delegate.triangles = static_cast<Vector3f *>(vertex_data);
    delegate.vertex_colour = static_cast<Vector3c *>(vertex_colour);

    dim3 thread(1024);
    dim3 block = dim3(div_up(state.num_total_hash_entries_, thread.x));

    select_blocks_coloured_kernel<<<block, thread>>>(delegate);

    safe_call(cudaMemcpy(&block_count, cuda_block_count, sizeof(uint), cudaMemcpyDeviceToHost));
    if (block_count == 0)
        return;

    thread = dim3(8, 8);
    block = dim3(div_up(block_count, 16), 16);

    generate_vertex_and_colour_array_kernel<<<block, thread>>>(delegate);

    safe_call(cudaMemcpy(&triangle_count, cuda_triangle_count, sizeof(uint), cudaMemcpyDeviceToHost));
    triangle_count = std::min(triangle_count, (uint)state.num_max_mesh_triangles_);

    safe_call(cudaFree(cuda_block_count));
    safe_call(cudaFree(cuda_triangle_count));
}

} // namespace cuda
} // namespace fusion