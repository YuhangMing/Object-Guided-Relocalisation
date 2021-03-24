#include "mapping/RayTraceEngine.h"
#include "mapping/VoxelStructUtils.h"
#include "mapping/ParallelScan.h"
#include "utils/safe_call.h"

#define RENDERING_BLOCK_SIZE_X 16
#define RENDERING_BLOCK_SIZE_Y 16
#define RENDERING_BLOCK_SUB_SAMPLE 8
#define MAX_RENDERING_BLOCK 100000

RayTraceEngine::RayTraceEngine(int w, int h, const Eigen::Matrix3f &K)
    : w(w), h(h)
{
    fx = K(0, 0);
    fy = K(1, 1);
    cx = K(0, 2);
    cy = K(1, 2);
    invfx = 1.0 / fx;
    invfy = 1.0 / fy;

    mTracedvmap.create(h, w, CV_32FC4);
    mTracedImage.create(h, w, CV_32FC4);
    mDepthMapMin.create(h / RENDERING_BLOCK_SUB_SAMPLE, w / RENDERING_BLOCK_SUB_SAMPLE, CV_32FC1);
    mDepthMapMax.create(h / RENDERING_BLOCK_SUB_SAMPLE, w / RENDERING_BLOCK_SUB_SAMPLE, CV_32FC1);

    safe_call(cudaMalloc(&mpNumVisibleBlocks, sizeof(uint) * 1));
    safe_call(cudaMalloc(&mpNumRenderingBlocks, sizeof(uint) * 1));
    safe_call(cudaMalloc(&mplRenderingBlockList, sizeof(RenderingBlock) * MAX_RENDERING_BLOCK));
}

RayTraceEngine::~RayTraceEngine()
{
    safe_call(cudaFree(mpNumVisibleBlocks));
    safe_call(cudaFree(mpNumRenderingBlocks));
    safe_call(cudaFree(mplRenderingBlockList));
}

uint RayTraceEngine::GetNumVisibleBlock()
{
    uint temp;
    safe_call(cudaMemcpy(&temp, mpNumVisibleBlocks, sizeof(uint), cudaMemcpyDeviceToHost));
    return temp;
}

uint RayTraceEngine::GetNumRenderingBlocks()
{
    uint temp;
    safe_call(cudaMemcpy(&temp, mpNumRenderingBlocks, sizeof(uint), cudaMemcpyDeviceToHost));
    return temp;
}

void RayTraceEngine::reset()
{
    safe_call(cudaMemset(mpNumVisibleBlocks, 0, sizeof(uint)));
    safe_call(cudaMemset(mpNumRenderingBlocks, 0, sizeof(uint)));
}

struct RenderingBlockDelegate
{
    Sophus::SE3f Tinv;
    int width, height;
    float fx, fy, cx, cy;
    float depthMin, depthMax;
    float voxelSize;

    uint *rendering_block_count;
    uint visible_block_count;

    HashEntry *visibleEntry;
    RayTraceEngine::RenderingBlock *listRenderingBlock;

    mutable cv::cuda::PtrStepSz<float> zRangeX;
    mutable cv::cuda::PtrStep<float> zRangeY;

    __device__ __forceinline__ bool createRenderingBlock(const Eigen::Vector3i &blockPos, RayTraceEngine::RenderingBlock &block) const
    {
        block.upper_left = Eigen::Matrix<short, 2, 1>(zRangeX.cols, zRangeX.rows);
        block.lower_right = Eigen::Matrix<short, 2, 1>(-1, -1);
        block.zrange = Eigen::Vector2f(depthMax, depthMin);

        float scale = voxelSize * BlockSize;
#pragma unroll
        for (int corner = 0; corner < 8; ++corner)
        {
            Eigen::Vector3i tmp = blockPos;
            tmp(0) += (corner & 1) ? 1 : 0;
            tmp(1) += (corner & 2) ? 1 : 0;
            tmp(2) += (corner & 4) ? 1 : 0;

            auto pt3d = Tinv * (tmp.cast<float>() * scale);
            Eigen::Vector2f pt2d = project(pt3d, fx, fy, cx, cy) / RENDERING_BLOCK_SUB_SAMPLE;

            if (block.upper_left(0) > std::floor(pt2d(0)))
                block.upper_left(0) = (int)std::floor(pt2d(0));
            if (block.lower_right(0) < ceil(pt2d(0)))
                block.lower_right(0) = (int)ceil(pt2d(0));
            if (block.upper_left(1) > std::floor(pt2d(1)))
                block.upper_left(1) = (int)std::floor(pt2d(1));
            if (block.lower_right(1) < ceil(pt2d(1)))
                block.lower_right(1) = (int)ceil(pt2d(1));
            if (block.zrange(0) > pt3d(2))
                block.zrange(0) = pt3d(2);
            if (block.zrange(1) < pt3d(2))
                block.zrange(1) = pt3d(2);
        }

        if (block.upper_left(0) < 0)
            block.upper_left(0) = 0;
        if (block.upper_left(1) < 0)
            block.upper_left(1) = 0;
        if (block.lower_right(0) >= zRangeX.cols)
            block.lower_right(0) = zRangeX.cols - 1;
        if (block.lower_right(1) >= zRangeX.rows)
            block.lower_right(1) = zRangeX.rows - 1;
        if (block.upper_left(0) > block.lower_right(0))
            return false;
        if (block.upper_left(1) > block.lower_right(1))
            return false;
        if (block.zrange(0) < depthMin)
            block.zrange(0) = depthMin;
        if (block.zrange(1) < depthMin)
            return false;

        return true;
    }

    __device__ __forceinline__ void splitRenderingBlock(int offset, const RayTraceEngine::RenderingBlock &block, int &nx, int &ny) const
    {
        for (int y = 0; y < ny; ++y)
        {
            for (int x = 0; x < nx; ++x)
            {
                if (offset < MAX_RENDERING_BLOCK)
                {
                    RayTraceEngine::RenderingBlock &b(listRenderingBlock[offset++]);
                    b.upper_left(0) = block.upper_left(0) + x * RENDERING_BLOCK_SIZE_X;
                    b.upper_left(1) = block.upper_left(1) + y * RENDERING_BLOCK_SIZE_Y;
                    b.lower_right(0) = block.upper_left(0) + (x + 1) * RENDERING_BLOCK_SIZE_X;
                    b.lower_right(1) = block.upper_left(1) + (y + 1) * RENDERING_BLOCK_SIZE_Y;
                    if (b.lower_right(0) > block.lower_right(0))
                        b.lower_right(0) = block.lower_right(0);
                    if (b.lower_right(1) > block.lower_right(1))
                        b.lower_right(1) = block.lower_right(1);
                    b.zrange = block.zrange;
                }
            }
        }
    }

    __device__ __forceinline__ void operator()() const
    {
        int x = threadIdx.x + blockDim.x * blockIdx.x;

        bool valid = false;
        uint requiredNoBlocks = 0;
        RayTraceEngine::RenderingBlock block;
        int nx, ny;

        if (x < visible_block_count && visibleEntry[x].ptr != -1)
        {
            valid = createRenderingBlock(visibleEntry[x].pos, block);
            float dx = (float)block.lower_right(0) - block.upper_left(0) + 1;
            float dy = (float)block.lower_right(1) - block.upper_left(1) + 1;
            nx = __float2int_ru(dx / RENDERING_BLOCK_SIZE_X);
            ny = __float2int_ru(dy / RENDERING_BLOCK_SIZE_Y);

            if (valid)
            {
                requiredNoBlocks = nx * ny;
                uint totalNoBlocks = *rendering_block_count + requiredNoBlocks;
                if (totalNoBlocks >= MAX_RENDERING_BLOCK)
                {
                    requiredNoBlocks = 0;
                }
            }
        }

        int offset = ParallelScan<1024>(requiredNoBlocks, rendering_block_count);
        if (valid && offset != -1 && (offset + requiredNoBlocks) < MAX_RENDERING_BLOCK)
            splitRenderingBlock(offset, block, nx, ny);
    }
};

struct FillRenderingBlockFunctor
{
    mutable cv::cuda::PtrStepSz<float> zRangeX;
    mutable cv::cuda::PtrStep<float> zRangeY;
    RayTraceEngine::RenderingBlock *listRenderingBlock;

    __device__ __forceinline__ void operator()() const
    {
        int x = threadIdx.x;
        int y = threadIdx.y;

        int block = blockIdx.x * 4 + blockIdx.y;
        if (block >= MAX_RENDERING_BLOCK)
            return;

        RayTraceEngine::RenderingBlock &b(listRenderingBlock[block]);

        int xpos = b.upper_left(0) + x;
        if (xpos > b.lower_right(0) || xpos >= zRangeX.cols)
            return;

        int ypos = b.upper_left(1) + y;
        if (ypos > b.lower_right(1) || ypos >= zRangeX.rows)
            return;

        atomicMin(&zRangeX.ptr(ypos)[xpos], b.zrange(0));
        atomicMax(&zRangeY.ptr(ypos)[xpos], b.zrange(1));

        return;
    }
};

void RayTraceEngine::UpdateRenderingBlocks(MapStruct *pMS, const Sophus::SE3d &Tcw)
{
    uint nBlocks = pMS->CheckNumVisibleBlocks(w, h, Tcw);
    if (nBlocks == 0)
        return;

    const int cols = mDepthMapMin.cols;
    const int rows = mDepthMapMin.rows;

    mDepthMapMin.setTo(cv::Scalar(100.f));
    mDepthMapMax.setTo(cv::Scalar(0));
    reset();

    RenderingBlockDelegate step1;

    step1.width = cols;
    step1.height = rows;
    step1.Tinv = Tcw.inverse().cast<float>();
    step1.zRangeX = mDepthMapMin;
    step1.zRangeY = mDepthMapMax;
    step1.fx = fx;
    step1.fy = fy;
    step1.cx = cx;
    step1.cy = cy;
    step1.visibleEntry = pMS->visibleTable;
    step1.visible_block_count = nBlocks;
    step1.rendering_block_count = mpNumRenderingBlocks;
    step1.listRenderingBlock = mplRenderingBlockList;
    step1.depthMax = 3.0f;
    step1.depthMin = 0.1f;
    step1.voxelSize = pMS->voxelSize;

    dim3 block = dim3(1024);
    dim3 grid = dim3(cv::divUp((size_t)nBlocks, block.x));
    call_device_functor<<<grid, block>>>(step1);

    nBlocks = GetNumRenderingBlocks();
    if (nBlocks == 0)
        return;

    FillRenderingBlockFunctor step2;
    step2.listRenderingBlock = mplRenderingBlockList;
    step2.zRangeX = mDepthMapMin;
    step2.zRangeY = mDepthMapMax;

    block = dim3(RENDERING_BLOCK_SIZE_X, RENDERING_BLOCK_SIZE_Y);
    grid = dim3((uint)ceil((float)nBlocks / 4), 4);
    call_device_functor<<<grid, block>>>(step2);
}

struct MapRenderingDelegate
{
    int width, height;
    mutable cv::cuda::PtrStep<Eigen::Vector4f> vmap;
    cv::cuda::PtrStepSz<float> zRangeX;
    cv::cuda::PtrStepSz<float> zRangeY;
    float invfx, invfy, cx, cy;
    Sophus::SE3f pose, Tinv;

    HashEntry *hashTable;
    Voxel *listBlock;
    int bucketSize;
    float voxelSizeInv;
    float raytraceStep;

    __device__ __forceinline__ float read_sdf(const Eigen::Vector3f &pt3d, bool &valid) const
    {
        Voxel *voxel = nullptr;
        findVoxel(floor(pt3d), voxel, hashTable, listBlock, bucketSize);
        if (voxel && voxel->wt != 0)
        {
            valid = true;
            return UnPackFloat(voxel->sdf);
        }
        else
        {
            valid = false;
            return 1.0;
        }
    }

    __device__ __forceinline__ float read_sdf_interped(const Eigen::Vector3f &pt, bool &valid) const
    {
        Eigen::Vector3f xyz = Eigen::Vector3f(pt(0) - floor(pt(0)), pt(1) - floor(pt(1)), pt(2) - floor(pt(2)));
        float sdf[2], result[4];
        bool valid_pt;

        sdf[0] = read_sdf(pt, valid_pt);
        sdf[1] = read_sdf(pt + Eigen::Vector3f(1, 0, 0), valid);
        valid_pt &= valid;
        result[0] = (1.0f - xyz(0)) * sdf[0] + xyz(0) * sdf[1];

        sdf[0] = read_sdf(pt + Eigen::Vector3f(0, 1, 0), valid);
        valid_pt &= valid;
        sdf[1] = read_sdf(pt + Eigen::Vector3f(1, 1, 0), valid);
        valid_pt &= valid;
        result[1] = (1.0f - xyz(0)) * sdf[0] + xyz(0) * sdf[1];
        result[2] = (1.0f - xyz(1)) * result[0] + xyz(1) * result[1];

        sdf[0] = read_sdf(pt + Eigen::Vector3f(0, 0, 1), valid);
        valid_pt &= valid;
        sdf[1] = read_sdf(pt + Eigen::Vector3f(1, 0, 1), valid);
        valid_pt &= valid;
        result[0] = (1.0f - xyz(0)) * sdf[0] + xyz(0) * sdf[1];

        sdf[0] = read_sdf(pt + Eigen::Vector3f(0, 1, 1), valid);
        valid_pt &= valid;
        sdf[1] = read_sdf(pt + Eigen::Vector3f(1, 1, 1), valid);
        valid_pt &= valid;
        result[1] = (1.0f - xyz(0)) * sdf[0] + xyz(0) * sdf[1];
        result[3] = (1.0f - xyz(1)) * result[0] + xyz(1) * result[1];
        valid = valid_pt;
        return (1.0f - xyz(2)) * result[2] + xyz(2) * result[3];
    }

    __device__ __forceinline__ void operator()() const
    {
        const int x = threadIdx.x + blockDim.x * blockIdx.x;
        const int y = threadIdx.y + blockDim.y * blockIdx.y;
        if (x >= width || y >= height)
            return;

        vmap.ptr(y)[x] = Eigen::Vector4f(0, 0, 0, -1.f);

        int u = __float2int_rd((float)x / 8);
        int v = __float2int_rd((float)y / 8);

        auto zNear = zRangeX.ptr(v)[u];
        auto zFar = zRangeY.ptr(v)[u];
        if (zNear < FLT_EPSILON || zFar < FLT_EPSILON ||
            isnan(zNear) || isnan(zFar))
            return;

        float sdf = 1.0f;
        float lastReadSDF;

        Eigen::Vector3f pt = UnProject(x, y, zNear, invfx, invfy, cx, cy);
        float dist_s = pt.norm() * voxelSizeInv;
        Eigen::Vector3f blockStart = pose * (pt)*voxelSizeInv;

        pt = UnProject(x, y, zFar, invfx, invfy, cx, cy);
        float distEnd = pt.norm() * voxelSizeInv;
        Eigen::Vector3f blockEnd = pose * (pt)*voxelSizeInv;

        Eigen::Vector3f dir = (blockEnd - blockStart).normalized();
        Eigen::Vector3f result = blockStart;

        bool sdfValid = false;
        bool ptFound = false;
        float step;

        while (dist_s < distEnd)
        {
            lastReadSDF = sdf;
            sdf = read_sdf(result, sdfValid);

            if (sdf <= 0.5f && sdf >= -0.5f)
                sdf = read_sdf_interped(result, sdfValid);
            if (sdf <= 0.0f)
                break;
            if (sdf >= 0.f && lastReadSDF < 0.f)
                return;
            if (sdfValid)
                step = max(sdf * raytraceStep, 1.0f);
            else
                step = 1;

            result += step * dir;
            dist_s += step;
        }

        if (sdf <= 0.0f)
        {
            step = sdf * raytraceStep;
            result += step * dir;

            sdf = read_sdf_interped(result, sdfValid);

            step = sdf * raytraceStep;
            result += step * dir;

            if (sdfValid)
                ptFound = true;
        }

        if (ptFound)
        {
            result = Tinv * (result / voxelSizeInv);
            vmap.ptr(y)[x].head<3>() = result;
            vmap.ptr(y)[x](3) = 1.f;
        }
    }
};

void RayTraceEngine::RayTrace(MapStruct *pMapStruct, const Sophus::SE3d &Tcm)
{
    if (pMapStruct)
    {
        uint nBlocks = pMapStruct->CheckNumVisibleBlocks(w, h, Tcm);
        if (nBlocks == 0)
            return;

        UpdateRenderingBlocks(pMapStruct, Tcm);
        nBlocks = GetNumRenderingBlocks();

        if (nBlocks == 0)
            return;

        MapRenderingDelegate delegate;

        delegate.width = w;
        delegate.height = h;
        delegate.vmap = mTracedvmap;
        delegate.zRangeX = mDepthMapMin;
        delegate.zRangeY = mDepthMapMax;
        delegate.invfx = invfx;
        delegate.invfy = invfy;
        delegate.cx = cx;
        delegate.cy = cy;
        delegate.pose = Tcm.cast<float>();
        delegate.Tinv = Tcm.inverse().cast<float>();
        delegate.hashTable = pMapStruct->mplHashTable;
        delegate.listBlock = pMapStruct->mplVoxelBlocks;
        delegate.bucketSize = pMapStruct->bucketSize;
        delegate.voxelSizeInv = 1.0 / pMapStruct->voxelSize;
        delegate.raytraceStep = pMapStruct->truncationDist / pMapStruct->voxelSize;

        dim3 block(4, 8);
        dim3 grid(cv::divUp(w, block.x), cv::divUp(h, block.y));

        call_device_functor<<<grid, block>>>(delegate);
    }
}

cv::cuda::GpuMat RayTraceEngine::GetVMap()
{
    return mTracedvmap;
}