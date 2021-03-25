#ifndef VOXEL_STRUCT_H
#define VOXEL_STRUCT_H

#include <iostream>
#include <mutex>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include "mapping/MeshEngine.h"
#include "mapping/RayTraceEngine.h"

#define BlockSize 8
#define BlockSize3 512
#define BlockSizeM1 7

class MeshEngine;
class RayTraceEngine;

struct HashEntry
{
    int ptr;
    int offset;
    Eigen::Vector3i pos;
};

struct Voxel
{
    short sdf;
    uchar wt;
};

struct PolygonMesh
{
    float *points;
    float *pointNormal;
    int nPoints;
};

class MapStruct
{
public:
    MapStruct(const Eigen::Matrix3f &K);
    void create(int num_hash, int bucket_size, int num_blocks,
                float voxel_size, float trunc_dist);

    Sophus::SE3d GetPose();
    void SetPose(const Sophus::SE3d &Tcw);
    void SetMeshEngine(MeshEngine *pMeshEngine);
    void SetTracer(RayTraceEngine *pRayTraceEngine);
    void reset();
    bool Empty();

    void Release();
    void Swap(MapStruct *pMapStruct);

    // FeatureMap fusion
    void ResetNumVisibleBlocks();
    uint GetNumVisibleBlocks();
    uint CheckNumVisibleBlocks(int cols, int rows, const Sophus::SE3d &Tcm);
    void Fuse(MapStruct *pMapStruct);
    void Fuse(cv::cuda::GpuMat depth, const Sophus::SE3d &Tcm);
    void Reserve(int hSize, int bSize, int vSize);

    void Hibernate();
    void ReActivate();
    bool mbInHibernation;

    void SetActiveFlag(bool flag);
    bool isActive();

public:
    void GenerateMesh();
    void DeleteMesh();

    float *mplPoint;
    float *mplNormal;
    int N;  // N = nTriangles*9
    bool mbHasMesh;

    // OpenGL buffer for Drawing
    float mColourTaint;
    uint mGlVertexBuffer;
    uint mGlNormalBuffer;
    bool mbVertexBufferCreated;

    // Mesh Engine
    MeshEngine *meshEngine;

    void RayTrace(const Sophus::SE3d &Tcm);
    cv::cuda::GpuMat GetRayTracingResult();
    cv::cuda::GpuMat GetRayTracingResultDepth();

    // RayTrace Engine
    RayTraceEngine *rayTracer;
    unsigned long int mnLastFusedFrameId;

    uint GetVisibleBlocks();
    void ResetVisibleBlocks();

public:
    size_t mnId;
    static size_t nNextId;
    std::mutex mutexDeviceMem;
    int *mplHeap, *mplHeapHib;
    int *mplHeapPtr, *mplHeapPtrHib;
    int *mplBucketMutex, *mplBucketMutexHib;

    HashEntry *mplHashTable, *mplHashTableHib;
    Voxel *mplVoxelBlocks, *mplVoxelBlocksHib;
    int *mpLinkedListHead, *mpLinkedListHeadHib;

    HashEntry *visibleTable;
    uint *visibleBlockNum;

    int bucketSize;
    int hashTableSize;
    int voxelBlockSize;
    float voxelSize;
    float truncationDist;

    Eigen::Matrix3f mK;

    bool mbActive;
    std::mutex mMutexActive;

    Sophus::SE3d mTcw;
    std::mutex mMutexPose;

    bool mbSubsumed;
    MapStruct *mpParent;
};

#endif