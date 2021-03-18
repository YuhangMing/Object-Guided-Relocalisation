#ifndef FUSION_VOXEL_HASHING_MAP_STRUCT
#define FUSION_VOXEL_HASHING_MAP_STRUCT

#include <iostream>
#include "macros.h"
#include "data_struct/voxel.h"
#include "data_struct/hash_entry.h"

#define BLOCK_SIZE 8
#define BLOCK_SIZE3 512
#define BLOCK_SIZE_SUB_1 7

namespace fusion
{

// Map info
class FUSION_EXPORT MapState
{
public:
    // The total number of buckets in the map
    // NOTE: buckets are allocated for each main entry
    // It dose not cover the excess entries
    int num_total_buckets_;

    // The total number of voxel blocks in the map
    // also determins the size of the heap memory
    // which is used for storing block addresses
    int num_total_voxel_blocks_;

    // The total number of hash entres in the map
    // This is a combination of main entries and
    // the excess entries
    int num_total_hash_entries_;

    int num_max_mesh_triangles_;
    int num_max_rendering_blocks_;

    float zmin_raycast;
    float zmax_raycast;
    float zmin_update;
    float zmax_update;
    float voxel_size;

    FUSION_HOST_AND_DEVICE int num_total_voxels() const;
    FUSION_HOST_AND_DEVICE int num_excess_entries() const;
    FUSION_HOST_AND_DEVICE int num_total_mesh_vertices() const;
    FUSION_HOST_AND_DEVICE float block_size_metric() const;
    FUSION_HOST_AND_DEVICE float inverse_voxel_size() const;
    FUSION_HOST_AND_DEVICE float truncation_dist() const;
    FUSION_HOST_AND_DEVICE float raycast_step_scale() const;
};

struct MapSize
{
    int num_blocks;
    int num_hash_entries;
    int num_buckets;
};

FUSION_DEVICE extern MapState param;

struct FUSION_EXPORT RenderingBlock
{
    Vector2s upper_left;
    Vector2s lower_right;
    Vector2f zrange;
};

struct FUSION_EXPORT MapStorage
{
    int *heap_mem_;
    int *excess_counter_;
    int *heap_mem_counter_;
    int *bucket_mutex_;
    Voxel *voxels_;
    HashEntry *hash_table_;
};

template <bool Device>
struct FUSION_EXPORT MapStruct
{
    FUSION_HOST MapStruct();
    FUSION_HOST MapStruct(MapState param);
    FUSION_HOST void create();
    FUSION_HOST void create(MapState param);
    FUSION_HOST void release();
    FUSION_HOST bool empty();
    FUSION_HOST void copyTo(MapStruct<Device> &) const;
    FUSION_HOST void upload(MapStruct<false> &);
    FUSION_HOST void download(MapStruct<false> &) const;
    FUSION_HOST void writeToDisk(std::string, bool binary = true) const;
    FUSION_HOST void exportModel(std::string) const;
    FUSION_HOST void readFromDisk(std::string, bool binary = true);
    FUSION_HOST void reset();

    MapStorage map;
    MapState state;
    MapSize size;
};

FUSION_DEVICE bool createHashEntry(MapStorage &map, const Vector3i &pos, const int &offset, HashEntry *entry);
FUSION_DEVICE bool deleteHashEntry(int *mem_counter, int *mem, int no_blocks, HashEntry &entry);
// FUSION_DEVICE bool deleteHashEntry(MapStorage &map, HashEntry &current);
FUSION_DEVICE void createBlock(MapStorage &map, const Vector3i &blockPos, int &bucket_index);
FUSION_DEVICE void deleteBlock(MapStorage &map, HashEntry &current);
FUSION_DEVICE void findVoxel(const MapStorage &map, const Vector3i &voxel_pos, Voxel *&out);
FUSION_DEVICE void findEntry(const MapStorage &map, const Vector3i &block_pos, HashEntry *&out);

//! Handy functions to modify the map
FUSION_HOST_AND_DEVICE int computeHash(const Vector3i &blockPos, const int &noBuckets);

//! Coordinate converters
FUSION_HOST_AND_DEVICE Vector3i worldPtToVoxelPos(Vector3f pt, const float &voxelSize);
FUSION_HOST_AND_DEVICE Vector3f voxelPosToWorldPt(const Vector3i &voxelPos, const float &voxelSize);
FUSION_HOST_AND_DEVICE Vector3i voxelPosToBlockPos(Vector3i voxelPos);
FUSION_HOST_AND_DEVICE Vector3i blockPosToVoxelPos(const Vector3i &blockPos);
FUSION_HOST_AND_DEVICE Vector3i voxelPosToLocalPos(Vector3i voxelPos);
FUSION_HOST_AND_DEVICE int localPosToLocalIdx(const Vector3i &localPos);
FUSION_HOST_AND_DEVICE Vector3i localIdxToLocalPos(const int &localIdx);
FUSION_HOST_AND_DEVICE int voxelPosToLocalIdx(const Vector3i &voxelPos);

} // namespace fusion

#endif