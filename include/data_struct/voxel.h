#ifndef FUSION_MAPPING_DATA_TYPES_H
#define FUSION_MAPPING_DATA_TYPES_H

#include "macros.h"
#include "math/matrix_type.h"
#include "math/vector_type.h"

namespace fusion
{

struct FUSION_EXPORT Voxel
{
    FUSION_HOST_AND_DEVICE inline Voxel();
    FUSION_HOST_AND_DEVICE inline Voxel(float sdf, float weight, Vector3c rgb, short label=0, short count=0);
    FUSION_HOST_AND_DEVICE inline float getSDF() const;
    FUSION_HOST_AND_DEVICE inline void setSDF(float val);
    FUSION_HOST_AND_DEVICE inline float getWeight() const;
    FUSION_HOST_AND_DEVICE inline void setWeight(float val);
    FUSION_HOST_AND_DEVICE inline unsigned char getLabel() const;

    short sdf;
    float weight;
    Vector3c rgb;

    // semantic info
    unsigned char label, count;
    unsigned char label_backup, count_backup;
    // Vector3c rgb_backup;
};

FUSION_HOST_AND_DEVICE inline Voxel::Voxel()
    : sdf(0), weight(0), rgb(0),
      label(0), count(0), 
      label_backup(0), count_backup(0)
{
}

FUSION_HOST_AND_DEVICE inline Voxel::Voxel(float sdf, float weight, Vector3c rgb, short label, short count)
    : weight(weight), rgb(rgb), label(label), count(count),
      label_backup(0), count_backup(0)
{
    setSDF(sdf);
}

FUSION_HOST_AND_DEVICE inline float unpackFloat(short val)
{
    return val / (float)32767;
}

FUSION_HOST_AND_DEVICE inline short packFloat(float val)
{
    return (short)(val * 32767);
}

FUSION_HOST_AND_DEVICE inline float Voxel::getSDF() const
{
    return unpackFloat(sdf);
}

FUSION_HOST_AND_DEVICE inline void Voxel::setSDF(float val)
{
    sdf = packFloat(val);
}

FUSION_HOST_AND_DEVICE inline float Voxel::getWeight() const
{
    return weight;
}

FUSION_HOST_AND_DEVICE inline void Voxel::setWeight(float val)
{
    weight = val;
}

FUSION_HOST_AND_DEVICE inline unsigned char Voxel::getLabel() const
{
    return label;
}

} // namespace fusion

#endif