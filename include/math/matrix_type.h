#ifndef FUSION_MATH_ROTATION_H
#define FUSION_MATH_ROTATION_H

#include "macros.h"
#include "math/vector_type.h"

namespace fusion
{

template <class T>
struct FUSION_EXPORT Matrix3x3
{
    Vector3<T> R0, R1, R2;

    FUSION_HOST_AND_DEVICE inline Matrix3x3() : R0(0), R1(0), R2(0) {}
    FUSION_HOST_AND_DEVICE inline Matrix3x3(const Matrix3x3<T> &M) : R0(M.R0), R1(M.R1), R2(M.R2) {}

#ifdef EIGEN_MACRO_H
    FUSION_HOST_AND_DEVICE inline Matrix3x3(const Eigen::Matrix<T, 3, 3> &M)
        : R0(M(0, 0), M(0, 1), M(0, 2)),
          R1(M(1, 0), M(1, 1), M(1, 2)),
          R2(M(2, 0), M(2, 1), M(2, 2))
    {
    }

    FUSION_HOST_AND_DEVICE inline Matrix3x3(const Eigen::Matrix<T, 4, 4> &M)
        : R0(M(0, 0), M(0, 1), M(0, 2)),
          R1(M(1, 0), M(1, 1), M(1, 2)),
          R2(M(2, 0), M(2, 1), M(2, 2))
    {
    }

#endif

    FUSION_HOST_AND_DEVICE inline Vector3<T> operator()(const Vector3<T> &V)
    {
        return Vector3<T>(R0 * V, R1 * V, R2 * V);
    }

    FUSION_HOST_AND_DEVICE inline Vector3<T> operator()(const Vector4<T> &V)
    {
        auto V3 = ToVector3(V);
        return Vector3<T>(R0 * V3, R1 * V3, R2 * V3);
    }
};

template <class T>
struct FUSION_EXPORT Matrix3x4
{
    Vector4<T> R0, R1, R2;

    FUSION_HOST_AND_DEVICE inline Matrix3x4() : R0(0), R1(0), R2(0) {}

#ifdef EIGEN_MACRO_H
    FUSION_HOST_AND_DEVICE inline Matrix3x4(const Eigen::Matrix<float, 3, 4> &M)
        : R0(M(0, 0), M(0, 1), M(0, 2), M(0, 3)),
          R1(M(1, 0), M(1, 1), M(1, 2), M(1, 3)),
          R2(M(2, 0), M(2, 1), M(2, 2), M(2, 3))
    {
    }

    FUSION_HOST_AND_DEVICE inline Matrix3x4(const Eigen::Matrix<float, 4, 4> &M)
        : R0(M(0, 0), M(0, 1), M(0, 2), M(0, 3)),
          R1(M(1, 0), M(1, 1), M(1, 2), M(1, 3)),
          R2(M(2, 0), M(2, 1), M(2, 2), M(2, 3))
    {
    }
#endif

#ifdef SOPHUS_SE3_HPP
    FUSION_HOST_AND_DEVICE inline Matrix3x4(const Sophus::Matrix<float, 3, 4> &M)
        : R0(M(0, 0), M(0, 1), M(0, 2), M(0, 3)),
          R1(M(1, 0), M(1, 1), M(1, 2), M(1, 3)),
          R2(M(2, 0), M(2, 1), M(2, 2), M(2, 3))
    {
    }

    FUSION_HOST_AND_DEVICE inline Matrix3x4(const Sophus::Matrix<float, 4, 4> &M)
        : R0(M(0, 0), M(0, 1), M(0, 2), M(0, 3)),
          R1(M(1, 0), M(1, 1), M(1, 2), M(1, 3)),
          R2(M(2, 0), M(2, 1), M(2, 2), M(2, 3))
    {
    }
#endif

    FUSION_HOST_AND_DEVICE inline Vector3<T> rotate(const Vector3<T> &V) const
    {
        return Vector3<T>(
            R0.x * V.x + R0.y * V.y + R0.z * V.z,
            R1.x * V.x + R1.y * V.y + R1.z * V.z,
            R2.x * V.x + R2.y * V.y + R2.z * V.z);
    }

    FUSION_HOST_AND_DEVICE inline Vector3<T> operator()(const Vector3<T> &V) const
    {
        Vector4<T> V4 = Vector4<T>(V, 1);
        return Vector3<T>(R0 * V4, R1 * V4, R2 * V4);
    }

    FUSION_HOST_AND_DEVICE inline Vector4<T> operator()(const Vector4<T> &V) const
    {
        return Vector4<T>(R0 * V, R1 * V, R2 * V, 1);
    }
};

using Matrix3x3f = Matrix3x3<float>;
using Matrix3x3d = Matrix3x3<double>;

using Matrix3x4f = Matrix3x4<float>;
using Matrix3x4d = Matrix3x4<double>;

} // namespace fusion

#endif