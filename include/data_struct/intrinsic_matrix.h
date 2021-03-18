#ifndef FUSION_CORE_INTRINSIC_MATRIX_H
#define FUSION_CORE_INTRINSIC_MATRIX_H

#include <memory>
#include <vector>
#include <iostream>
#include "macros.h"

namespace fusion
{

struct FUSION_EXPORT IntrinsicMatrix
{
    FUSION_HOST_AND_DEVICE inline IntrinsicMatrix();
    FUSION_HOST_AND_DEVICE inline IntrinsicMatrix(int cols, int rows, float fx, float fy, float cx, float cy);
    FUSION_HOST_AND_DEVICE inline IntrinsicMatrix pyr_down() const;

    int width, height;
    float fx, fy, cx, cy, invfx, invfy;
};

FUSION_HOST_AND_DEVICE inline IntrinsicMatrix::IntrinsicMatrix()
    : width(0), height(0), fx(0), fy(0), cx(0), cy(0), invfx(0), invfy(0)
{
}

FUSION_HOST_AND_DEVICE inline IntrinsicMatrix::IntrinsicMatrix(
    int cols, int rows, float fx, float fy, float cx, float cy)
    : width(cols), height(rows), fx(fx), fy(fy), cx(cx), cy(cy), invfx(1.0f / fx), invfy(1.0f / fy)
{
}

FUSION_HOST_AND_DEVICE inline IntrinsicMatrix IntrinsicMatrix::pyr_down() const
{
    float s = 0.5f;
    return IntrinsicMatrix(s * width, s * height, s * fx, s * fy, s * cx, s * cy);
}

FUSION_HOST inline void BuildIntrinsicPyramid(IntrinsicMatrix base_K, std::vector<IntrinsicMatrix> &pyr, const int NUM_PYR)
{
    pyr.clear();
    pyr.push_back(base_K);
    for (int i = 0; i < NUM_PYR - 1; ++i)
    {
        pyr.push_back(pyr[i].pyr_down());
    }
}

} // namespace fusion

#endif
