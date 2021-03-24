#include "mapping/MeshEngine.h"
#include "mapping/ParallelScan.h"
#include "mapping/TriangleTable.h"
#include "mapping/VoxelStructUtils.h"
#include "utils/safe_call.h"

struct CollectBlocksFunctor
{
    uint *mpNumEntry;
    int hashTableSize;

    HashEntry *mplEntry;
    HashEntry *mplHashTable;

    __device__ __forceinline__ void operator()() const;
};

__device__ __forceinline__ void CollectBlocksFunctor::operator()() const
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ bool needScan;

    if (x == 0)
        needScan = false;

    __syncthreads();

    uint val = 0;
    if (x < hashTableSize && mplHashTable[x].ptr != -1)
    {
        needScan = true;
        val = 1;
    }

    __syncthreads();

    if (needScan)
    {
        int offset = ParallelScan<1024>(val, mpNumEntry);
        if (offset != -1 && offset < hashTableSize)
            mplEntry[offset] = mplHashTable[x];
    }
}

struct MeshificationFunctor
{
    int hashTableSize;
    int bucketSize;
    float voxelSize;
    size_t bufferSize;

    HashEntry *mplEntry;
    HashEntry *mplHashTable;
    Voxel *mplVoxelBlocks;

    uint *mpNumEntry;
    uint *mpNumTriangle;

    float *mplNormals;
    float *mplTriangle;

    __device__ __forceinline__ float read_sdf(Eigen::Vector3f pt, bool &valid) const
    {
        Voxel *voxel = nullptr;
        findVoxel(floor(pt), voxel, mplHashTable, mplVoxelBlocks, bucketSize);
        if (voxel && voxel->wt != 0)
        {
            valid = true;
            return UnPackFloat(voxel->sdf);
        }
        else
        {
            valid = false;
            return 1.0f;
        }
    }

    __device__ __forceinline__ bool read_sdf_list(float *sdf, Eigen::Vector3f pos) const
    {
        bool valid = false;
        sdf[0] = read_sdf(pos, valid);
        if (!valid)
            return false;

        sdf[1] = read_sdf(pos + Eigen::Vector3f(1, 0, 0), valid);
        if (!valid)
            return false;

        sdf[2] = read_sdf(pos + Eigen::Vector3f(1, 1, 0), valid);
        if (!valid)
            return false;

        sdf[3] = read_sdf(pos + Eigen::Vector3f(0, 1, 0), valid);
        if (!valid)
            return false;

        sdf[4] = read_sdf(pos + Eigen::Vector3f(0, 0, 1), valid);
        if (!valid)
            return false;

        sdf[5] = read_sdf(pos + Eigen::Vector3f(1, 0, 1), valid);
        if (!valid)
            return false;

        sdf[6] = read_sdf(pos + Eigen::Vector3f(1, 1, 1), valid);
        if (!valid)
            return false;

        sdf[7] = read_sdf(pos + Eigen::Vector3f(0, 1, 1), valid);
        if (!valid)
            return false;

        return true;
    }

    __device__ __forceinline__ float InterpolateSDF(float &v1, float &v2) const
    {
        if (fabs(0 - v1) < 1e-6)
            return 0;
        if (fabs(0 - v2) < 1e-6)
            return 1;
        if (fabs(v1 - v2) < 1e-6)
            return 0;
        return (0 - v1) / (v2 - v1);
    }

    __device__ __forceinline__ int MakeVertList(Eigen::Vector3f *verts, const Eigen::Vector3f &pos) const
    {
        float sdf[8];

        if (!read_sdf_list(sdf, pos))
            return -1;

        int CubeIdx = 0;
        if (sdf[0] < 0)
            CubeIdx |= 1;
        if (sdf[1] < 0)
            CubeIdx |= 2;
        if (sdf[2] < 0)
            CubeIdx |= 4;
        if (sdf[3] < 0)
            CubeIdx |= 8;
        if (sdf[4] < 0)
            CubeIdx |= 16;
        if (sdf[5] < 0)
            CubeIdx |= 32;
        if (sdf[6] < 0)
            CubeIdx |= 64;
        if (sdf[7] < 0)
            CubeIdx |= 128;

        if (edgeTable[CubeIdx] == 0)
            return -1;

        if (edgeTable[CubeIdx] & 1)
        {
            float val = InterpolateSDF(sdf[0], sdf[1]);
            verts[0] = pos + Eigen::Vector3f(val, 0, 0);
        }

        if (edgeTable[CubeIdx] & 2)
        {
            float val = InterpolateSDF(sdf[1], sdf[2]);
            verts[1] = pos + Eigen::Vector3f(1, val, 0);
        }

        if (edgeTable[CubeIdx] & 4)
        {
            float val = InterpolateSDF(sdf[2], sdf[3]);
            verts[2] = pos + Eigen::Vector3f(1 - val, 1, 0);
        }

        if (edgeTable[CubeIdx] & 8)
        {
            float val = InterpolateSDF(sdf[3], sdf[0]);
            verts[3] = pos + Eigen::Vector3f(0, 1 - val, 0);
        }

        if (edgeTable[CubeIdx] & 16)
        {
            float val = InterpolateSDF(sdf[4], sdf[5]);
            verts[4] = pos + Eigen::Vector3f(val, 0, 1);
        }

        if (edgeTable[CubeIdx] & 32)
        {
            float val = InterpolateSDF(sdf[5], sdf[6]);
            verts[5] = pos + Eigen::Vector3f(1, val, 1);
        }

        if (edgeTable[CubeIdx] & 64)
        {
            float val = InterpolateSDF(sdf[6], sdf[7]);
            verts[6] = pos + Eigen::Vector3f(1 - val, 1, 1);
        }

        if (edgeTable[CubeIdx] & 128)
        {
            float val = InterpolateSDF(sdf[7], sdf[4]);
            verts[7] = pos + Eigen::Vector3f(0, 1 - val, 1);
        }

        if (edgeTable[CubeIdx] & 256)
        {
            float val = InterpolateSDF(sdf[0], sdf[4]);
            verts[8] = pos + Eigen::Vector3f(0, 0, val);
        }

        if (edgeTable[CubeIdx] & 512)
        {
            float val = InterpolateSDF(sdf[1], sdf[5]);
            verts[9] = pos + Eigen::Vector3f(1, 0, val);
        }

        if (edgeTable[CubeIdx] & 1024)
        {
            float val = InterpolateSDF(sdf[2], sdf[6]);
            verts[10] = pos + Eigen::Vector3f(1, 1, val);
        }

        if (edgeTable[CubeIdx] & 2048)
        {
            float val = InterpolateSDF(sdf[3], sdf[7]);
            verts[11] = pos + Eigen::Vector3f(0, 1, val);
        }

        return CubeIdx;
    }

    __device__ __forceinline__ void operator()() const
    {
        int x = blockIdx.y * gridDim.x + blockIdx.x;
        if (*mpNumTriangle >= bufferSize || x >= *mpNumEntry)
            return;

        Eigen::Vector3f verts[12];
        Eigen::Vector3i pos = mplEntry[x].pos * BlockSize;

        for (int z = 0; z < BlockSize; ++z)
        {
            Eigen::Vector3i LocalPos = Eigen::Vector3i(threadIdx.x, threadIdx.y, z);
            int CubeIdx = MakeVertList(verts, (pos + LocalPos).cast<float>());
            if (CubeIdx <= 0)
                continue;

            for (int i = 0; triTable[CubeIdx][i] != -1; i += 3)
            {
                uint TriangleIdx = atomicAdd(mpNumTriangle, 1);
                if (TriangleIdx < bufferSize)
                {
                    Eigen::Vector3f vert0 = verts[triTable[CubeIdx][i]] * voxelSize;
                    Eigen::Vector3f vert1 = verts[triTable[CubeIdx][i + 1]] * voxelSize;
                    Eigen::Vector3f vert2 = verts[triTable[CubeIdx][i + 2]] * voxelSize;
                    mplTriangle[TriangleIdx * 9] = vert0(0);
                    mplTriangle[TriangleIdx * 9 + 1] = vert0(1);
                    mplTriangle[TriangleIdx * 9 + 2] = vert0(2);
                    mplTriangle[TriangleIdx * 9 + 3] = vert1(0);
                    mplTriangle[TriangleIdx * 9 + 4] = vert1(1);
                    mplTriangle[TriangleIdx * 9 + 5] = vert1(2);
                    mplTriangle[TriangleIdx * 9 + 6] = vert2(0);
                    mplTriangle[TriangleIdx * 9 + 7] = vert2(1);
                    mplTriangle[TriangleIdx * 9 + 8] = vert2(2);
                    Eigen::Vector3f normal = (vert1 - vert0).cross(vert2 - vert0).normalized();
                    mplNormals[TriangleIdx * 9] = mplNormals[TriangleIdx * 9 + 3] = mplNormals[TriangleIdx * 9 + 6] = normal(0);
                    mplNormals[TriangleIdx * 9 + 1] = mplNormals[TriangleIdx * 9 + 4] = mplNormals[TriangleIdx * 9 + 7] = normal(1);
                    mplNormals[TriangleIdx * 9 + 2] = mplNormals[TriangleIdx * 9 + 5] = mplNormals[TriangleIdx * 9 + 8] = normal(2);
                }
            }
        }
    }
};

MeshEngine::MeshEngine(int MaxNumTriangle) : mMaxNumTriangle(MaxNumTriangle)
{
    safe_call(cudaMalloc((void **)&cuda_block_count, sizeof(uint)));
    safe_call(cudaMalloc((void **)&cuda_triangle_count, sizeof(uint)));
    safe_call(cudaMalloc((void **)&mplVertexBuffer, sizeof(float) * 3 * MaxNumTriangle));
    safe_call(cudaMalloc((void **)&mplNormalBuffer, sizeof(float) * 3 * MaxNumTriangle));
}

MeshEngine::~MeshEngine()
{
    safe_call(cudaFree(cuda_block_count));
    safe_call(cudaFree(cuda_triangle_count));
    safe_call(cudaFree(mplVertexBuffer));
    safe_call(cudaFree(mplNormalBuffer));
}

void MeshEngine::ResetTemporaryValues()
{
    safe_call(cudaMemset(cuda_block_count, 0, sizeof(uint)));
    safe_call(cudaMemset(cuda_triangle_count, 0, sizeof(uint)));
}

void MeshEngine::Meshify(MapStruct *pMapStruct)
{
    ResetTemporaryValues();

    CollectBlocksFunctor functor1;
    functor1.mpNumEntry = cuda_block_count;
    functor1.mplEntry = pMapStruct->visibleTable;
    functor1.mplHashTable = pMapStruct->mplHashTable;
    functor1.hashTableSize = pMapStruct->hashTableSize;

    dim3 block(1024);
    dim3 grid(cv::divUp(pMapStruct->hashTableSize, block.x));
    call_device_functor<<<grid, block>>>(functor1);

    uint nHashEntry = 0;
    safe_call(cudaMemcpy(&nHashEntry, cuda_block_count, sizeof(uint), cudaMemcpyDeviceToHost));

    if (nHashEntry == 0)
        return;

    MeshificationFunctor functor2;
    functor2.mplEntry = pMapStruct->visibleTable;
    functor2.mpNumEntry = cuda_block_count;
    functor2.mpNumTriangle = cuda_triangle_count;
    functor2.mplTriangle = mplVertexBuffer;
    functor2.mplNormals = mplNormalBuffer;
    functor2.mplHashTable = pMapStruct->mplHashTable;
    functor2.mplVoxelBlocks = pMapStruct->mplVoxelBlocks;
    functor2.hashTableSize = pMapStruct->hashTableSize;
    functor2.bucketSize = pMapStruct->bucketSize;
    functor2.voxelSize = pMapStruct->voxelSize;
    functor2.bufferSize = mMaxNumTriangle;

    block = dim3(8, 8);
    grid = dim3(cv::divUp((size_t)nHashEntry, 16U), 16U);
    call_device_functor<<<grid, block>>>(functor2);

    uint nTriangles = 0;
    safe_call(cudaMemcpy(&nTriangles, cuda_triangle_count, sizeof(uint), cudaMemcpyDeviceToHost));
    nTriangles = std::min(nTriangles, (uint)mMaxNumTriangle);

    pMapStruct->mplPoint = new float[nTriangles * 9];
    pMapStruct->mplNormal = new float[nTriangles * 9];
    pMapStruct->N = nTriangles * 9;

    safe_call(cudaMemcpy(pMapStruct->mplPoint, mplVertexBuffer, sizeof(float) * 9 * nTriangles, cudaMemcpyDeviceToHost));
    safe_call(cudaMemcpy(pMapStruct->mplNormal, mplNormalBuffer, sizeof(float) * 9 * nTriangles, cudaMemcpyDeviceToHost));
}