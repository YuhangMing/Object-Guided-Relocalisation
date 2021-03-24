#ifndef VOXEL_STRUCT_UTILS_H
#define VOXEL_STRUCT_UTILS_H

#include "mapping/VoxelMap.h"

__device__ __forceinline__ float UnPackFloat(short val)
{
    return val / (float)SHRT_MAX;
}

__device__ __forceinline__ short PackFloat(float val)
{
    return (short)(val * SHRT_MAX);
}

__device__ __forceinline__ int hash(const Eigen::Vector3i &pos, const int &N)
{
    int res = (pos(0) * 73856093) ^ (pos(1) * 19349669) ^ (pos(2) * 83492791);
    res %= N;
    return res < 0 ? res + N : res;
}

__device__ __forceinline__ Eigen::Vector3i floor(const Eigen::Vector3f &pt)
{
    return Eigen::Vector3i((int)floor(pt(0)), (int)floor(pt(1)), (int)floor(pt(2)));
}

__device__ __forceinline__ Eigen::Vector3i WorldPtToVoxelPos(Eigen::Vector3f pt, const float &voxelSize)
{
    pt = pt / voxelSize;
    return floor(pt);
}

__host__ __device__ __forceinline__ Eigen::Vector3f VoxelPosToWorldPt(const Eigen::Vector3i &voxelPos, const float &voxelSize)
{
    return voxelPos.cast<float>() * voxelSize;
}

__device__ __forceinline__ Eigen::Vector3i VoxelPosToBlockPos(Eigen::Vector3i voxelPos)
{
    if (voxelPos(0) < 0)
        voxelPos(0) -= BlockSizeM1;
    if (voxelPos(1) < 0)
        voxelPos(1) -= BlockSizeM1;
    if (voxelPos(2) < 0)
        voxelPos(2) -= BlockSizeM1;

    return voxelPos / BlockSize;
}

__host__ __device__ __forceinline__ Eigen::Vector3i BlockPosToVoxelPos(const Eigen::Vector3i &blockPos)
{
    return blockPos * BlockSize;
}

__device__ __forceinline__ Eigen::Vector3i VoxelPosToLocalPos(Eigen::Vector3i voxelPos)
{
    int x = voxelPos(0) % BlockSize;
    int y = voxelPos(1) % BlockSize;
    int z = voxelPos(2) % BlockSize;

    if (x < 0)
        x += BlockSize;
    if (y < 0)
        y += BlockSize;
    if (z < 0)
        z += BlockSize;

    return Eigen::Vector3i(x, y, z);
}

__device__ __forceinline__ int LocalPosToLocalIdx(const Eigen::Vector3i &localPos)
{
    return localPos(2) * BlockSize * BlockSize + localPos(1) * BlockSize + localPos(0);
}

__device__ __forceinline__ Eigen::Vector3i LocalIdxToLocalPos(const int &localIdx)
{
    uint x = localIdx % BlockSize;
    uint y = localIdx % (BlockSize * BlockSize) / BlockSize;
    uint z = localIdx / (BlockSize * BlockSize);
    return Eigen::Vector3i(x, y, z);
}

__device__ __forceinline__ int VoxelPosToLocalIdx(const Eigen::Vector3i &voxelPos)
{
    return LocalPosToLocalIdx(VoxelPosToLocalPos(voxelPos));
}

__device__ __forceinline__ Eigen::Vector2f project(const Eigen::Vector3f &pt,
                                                   const float &fx, const float &fy,
                                                   const float &cx, const float &cy)
{
    return Eigen::Vector2f(fx * pt(0) / pt(2) + cx, fy * pt(1) / pt(2) + cy);
}

__device__ __forceinline__ Eigen::Vector3f UnProject(const int &x, const int &y, const float &z,
                                                     const float &invfx, const float &invfy,
                                                     const float &cx, const float &cy)
{
    return Eigen::Vector3f(invfx * (x - cx) * z, invfy * (y - cy) * z, z);
}

__device__ __forceinline__ Eigen::Vector3f UnProjectWorld(const int &x, const int &y, const float &z,
                                                          const float &invfx, const float &invfy,
                                                          const float &cx, const float &cy,
                                                          const Sophus::SE3f &T)
{
    return T * UnProject(x, y, z, invfx, invfy, cx, cy);
}

__device__ __forceinline__ bool CheckPointVisibility(Eigen::Vector3f pt, const Sophus::SE3f &Tinv,
                                                     const int &cols, const int &rows,
                                                     const float &fx, const float &fy,
                                                     const float &cx, const float &cy,
                                                     const float &depthMin, const float &depthMax)
{
    pt = Tinv * pt;
    Eigen::Vector2f pt2d = project(pt, fx, fy, cx, cy);
    return !(pt2d(0) < 0 || pt2d(1) < 0 || pt2d(0) > cols - 1 || pt2d(1) > rows - 1 || pt(2) < depthMin || pt(2) > depthMax);
}

__device__ __forceinline__ bool CheckBlockVisibility(const Eigen::Vector3i &blockPos,
                                                     const Sophus::SE3f &Tinv,
                                                     const float &voxelSize,
                                                     const int &cols, const int &rows,
                                                     const float &fx, const float &fy,
                                                     const float &cx, const float &cy,
                                                     const float &depthMin, const float &depthMax)
{
    float scale = voxelSize * BlockSize;
#pragma unroll
    for (int corner = 0; corner < 8; ++corner)
    {
        Eigen::Vector3i tmp = blockPos;
        tmp(0) += (corner & 1) ? 1 : 0;
        tmp(1) += (corner & 2) ? 1 : 0;
        tmp(2) += (corner & 4) ? 1 : 0;

        if (CheckPointVisibility(tmp.cast<float>() * scale, Tinv,
                                 cols, rows, fx, fy, cx, cy,
                                 depthMin, depthMax))
            return true;
    }

    return false;
}

// compare val with the old value stored in *add and write the bigger one to *add
__device__ __forceinline__ void atomicMax(float *add, float val)
{
    int *address_as_i = (int *)add;
    int old = *address_as_i, assumed;
    do
    {
        assumed = old;
        old = atomicCAS(address_as_i, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
}

// compare val with the old value stored in *add and write the smaller one to *add
__device__ __forceinline__ void atomicMin(float *add, float val)
{
    int *address_as_i = (int *)add;
    int old = *address_as_i, assumed;
    do
    {
        assumed = old;
        old = atomicCAS(address_as_i, assumed, __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
}

__device__ __forceinline__ bool LockBucket(int *mutex)
{
    if (atomicExch(mutex, 1) != 1)
        return true;
    else
        return false;
}

__device__ __forceinline__ void UnLockBucket(int *mutex)
{
    atomicExch(mutex, 0);
}

__device__ __forceinline__ bool RemoveHashEntry(int *mplHeapPtr, int *mplHeap,
                                                const int &voxelBlockSize, HashEntry *pEntry)
{
    int old = atomicAdd(mplHeapPtr, 1);
    if (old < voxelBlockSize)
    {
        mplHeap[old + 1] = pEntry->ptr / BlockSize3;
        pEntry->ptr = -1;
        return true;
    }
    else
    {
        atomicSub(mplHeapPtr, 1);
        return false;
    }
}

__device__ __forceinline__ bool CreateHashEntry(int *mplHeap, int *mplHeapPtr, const Eigen::Vector3i &pos,
                                                const int &offset, HashEntry *pEntry)
{
    int old = atomicSub(mplHeapPtr, 1);
    if (old > 0)
    {
        int ptr = mplHeap[old];
        if (ptr != -1 && pEntry != nullptr)
        {
            pEntry->pos = pos;
            pEntry->ptr = ptr * BlockSize3;
            pEntry->offset = offset;
            return true;
        }
    }
    else
    {
        atomicAdd(mplHeapPtr, 1);
    }

    return false;
}

__device__ __forceinline__ HashEntry *CreateNewBlock(const Eigen::Vector3i &blockPos, int *mplHeap,
                                                     int *mplHeapPtr, HashEntry *mplHashTable, int *mplBucketMutex,
                                                     int *mpLinkedListHead, int hashTableSize, int bucketSize)
{
    auto volatileIdx = hash(blockPos, bucketSize);
    int *mutex = &mplBucketMutex[volatileIdx];
    HashEntry *current = &mplHashTable[volatileIdx];
    HashEntry *emptyEntry = nullptr;
    if (current->pos == blockPos && current->ptr != -1)
        return current;

    if (current->ptr == -1)
        emptyEntry = current;

    while (current->offset >= 0)
    {
        volatileIdx = bucketSize + current->offset - 1;
        current = &mplHashTable[volatileIdx];
        if (current->pos == blockPos && current->ptr != -1)
            return current;

        if (current->ptr == -1 && !emptyEntry)
            emptyEntry = current;
    }

    if (emptyEntry != nullptr)
    {
        if (LockBucket(mutex))
        {
            CreateHashEntry(mplHeap, mplHeapPtr, blockPos, current->offset, emptyEntry);
            UnLockBucket(mutex);
            return emptyEntry;
        }
    }
    else
    {
        if (LockBucket(mutex))
        {
            int offset = atomicAdd(mpLinkedListHead, 1);
            if ((offset + bucketSize) < hashTableSize)
            {
                emptyEntry = &mplHashTable[bucketSize + offset - 1];
                if (CreateHashEntry(mplHeap, mplHeapPtr, blockPos, -1, emptyEntry))
                    current->offset = offset;
            }
            else
                atomicSub(mpLinkedListHead, 1);

            UnLockBucket(mutex);
            return emptyEntry;
        }
    }

    return nullptr;
}

__device__ __forceinline__ bool findEntry(const Eigen::Vector3i &blockPos, HashEntry *&out,
                                          HashEntry *mplHashTable, int bucketSize)
{
    uint volatileIdx = hash(blockPos, bucketSize);
    out = &mplHashTable[volatileIdx];
    if (out->ptr != -1 && out->pos == blockPos)
        return true;

    while (out->offset >= 0)
    {
        volatileIdx = bucketSize + out->offset - 1;
        out = &mplHashTable[volatileIdx];
        if (out->ptr != -1 && out->pos == blockPos)
            return true;
    }

    out = nullptr;
    return false;
}

__device__ __forceinline__ void findVoxel(const Eigen::Vector3i &voxelPos, Voxel *&pVoxel,
                                          HashEntry *plHashTable, Voxel *pListBlocks, int bucketSize)
{
    HashEntry *pEntry;
    if (findEntry(VoxelPosToBlockPos(voxelPos), pEntry, plHashTable, bucketSize))
        pVoxel = &pListBlocks[pEntry->ptr + VoxelPosToLocalIdx(voxelPos)];
}

__device__ __forceinline__ void findVoxel(const Eigen::Vector3f &worldPos, Voxel *&pVoxel, float voxelSize,
                                          HashEntry *plHashTable, Voxel *pListBlocks, int bucketSize)
{
    HashEntry *pEntry;
    Eigen::Vector3i voxelPos = WorldPtToVoxelPos(worldPos, voxelSize);
    if (findEntry(VoxelPosToBlockPos(voxelPos), pEntry, plHashTable, bucketSize))
        pVoxel = &pListBlocks[pEntry->ptr + VoxelPosToLocalIdx(voxelPos)];
}

#endif