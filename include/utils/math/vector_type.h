#ifndef FUSION_MATH_TYPES_H
#define FUSION_MATH_TYPES_H

#include <cmath>
#include <sophus/se3.hpp>
#include "utils/macros.h"

namespace fusion
{

using uchar = unsigned char;

//! 2-dimensional vector type
template <class T>
struct Vector2
{
    FUSION_HOST_AND_DEVICE inline Vector2() : x(0), y(0) {}
    FUSION_HOST_AND_DEVICE inline Vector2(const Vector2<T> &other) : x(other.x), y(other.y) {}
    FUSION_HOST_AND_DEVICE inline Vector2(T x, T y) : x(x), y(y) {}
    FUSION_HOST_AND_DEVICE inline Vector2(T val) : x(val), y(val) {}

    FUSION_HOST_AND_DEVICE inline Vector2<T> operator+(const Vector2<T> &other) const
    {
        return Vector2<T>(x + other.x, y + other.y);
    }

    FUSION_HOST_AND_DEVICE inline Vector2<T> operator-(const Vector2<T> &other) const
    {
        return Vector2<T>(x - other.x, y - other.y);
    }

    FUSION_HOST_AND_DEVICE inline T dot(const Vector2<T> &V) const
    {
        return x * V.x + y * V.y;
    }

    FUSION_HOST_AND_DEVICE inline T operator*(const Vector2<T> &other) const
    {
        return x * other.x + y * other.y;
    }

    FUSION_HOST_AND_DEVICE inline Vector2<T> operator/(const T val) const
    {
        return Vector2<T>(x / val, y / val);
    }

    T x, y;
};

//! 3-dimensional vector type
template <class T>
struct Vector3
{
    FUSION_HOST_AND_DEVICE inline Vector3() : x(0), y(0), z(0) {}
    FUSION_HOST_AND_DEVICE inline Vector3(const Vector3<T> &other) : x(other.x), y(other.y), z(other.z) {}
    FUSION_HOST_AND_DEVICE inline Vector3(T x, T y, T z) : x(x), y(y), z(z) {}
    FUSION_HOST_AND_DEVICE inline Vector3(T x, T y) : x(x), y(y), z(1) {}
    FUSION_HOST_AND_DEVICE inline Vector3(T val) : x(val), y(val), z(val) {}

    FUSION_HOST_AND_DEVICE inline bool operator==(const Vector3<T> &other) const
    {
        return x == other.x && y == other.y && z == other.z;
    }

    FUSION_HOST_AND_DEVICE inline Vector3<T> operator+=(const Vector3<T> &other)
    {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    FUSION_HOST_AND_DEVICE inline Vector3<T> operator+(const Vector3<T> &other) const
    {
        return Vector3<T>(x + other.x, y + other.y, z + other.z);
    }

    FUSION_HOST_AND_DEVICE inline Vector3<T> operator-(const Vector3<T> &other) const
    {
        return Vector3<T>(x - other.x, y - other.y, z - other.z);
    }

    FUSION_HOST_AND_DEVICE inline T operator*(const Vector3<T> &other) const
    {
        return x * other.x + y * other.y + z * other.z;
    }

    FUSION_HOST_AND_DEVICE inline Vector3<T> operator/(const T val) const
    {
        return Vector3<T>(x / val, y / val, z / val);
    }

    FUSION_HOST_AND_DEVICE inline Vector3<T> operator%(const T val) const
    {
        return Vector3<T>(x % val, y % val, z % val);
    }

    FUSION_HOST_AND_DEVICE inline float norm() const
    {
        return sqrt((float)(x * x + y * y + z * z));
    }

    FUSION_HOST_AND_DEVICE inline T dot(const Vector3<T> &other) const
    {
        return x * other.x + y * other.y + z * other.z;
    }

    FUSION_HOST_AND_DEVICE inline Vector3<T> cross(const Vector3<T> &other) const
    {
        return Vector3<T>(y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x);
    }

    T x, y, z;
};

//! 4-dimensional vector type
template <class T>
struct Vector4
{
    FUSION_HOST_AND_DEVICE inline Vector4() : x(0), y(0), z(0), w(0) {}
    FUSION_HOST_AND_DEVICE inline Vector4(const Vector3<T> &other) : x(other.x), y(other.y), z(other.z), w(other.w) {}
    FUSION_HOST_AND_DEVICE inline Vector4(T x, T y, T z, T w) : x(x), y(y), z(z), w(w) {}
    FUSION_HOST_AND_DEVICE inline Vector4(T val) : x(val), y(val), z(val), w(val) {}
    FUSION_HOST_AND_DEVICE inline Vector4(const Vector3<T> &other, const T &val) : x(other.x), y(other.y), z(other.z), w(val) {}

    FUSION_HOST_AND_DEVICE inline Vector4<T> operator+(const Vector4<T> &other) const
    {
        return Vector4<T>(x + other.x, y + other.y, z + other.z, w + other.w);
    }

    FUSION_HOST_AND_DEVICE inline Vector4<T> operator-(const Vector4<T> &other) const
    {
        return Vector4<T>(x - other.x, y - other.y, z - other.z, w - other.w);
    }

    FUSION_HOST_AND_DEVICE inline T operator*(const Vector4<T> &other) const
    {
        return x * other.x + y * other.y + z * other.z + w * other.w;
    }

    FUSION_HOST_AND_DEVICE inline Vector4<T> operator/(const T val) const
    {
        return Vector4<T>(x / val, y / val, z / val, w / val);
    }

    T x, y, z, w;
};

//! Vector aliases
using Vector2i = Vector2<int>;
using Vector2s = Vector2<short>;
using Vector2f = Vector2<float>;
using Vector2c = Vector2<uchar>;
using Vector2d = Vector2<double>;

using Vector3i = Vector3<int>;
using Vector3f = Vector3<float>;
using Vector3c = Vector3<uchar>;
using Vector3d = Vector3<double>;

using Vector4i = Vector4<int>;
using Vector4f = Vector4<float>;
using Vector4c = Vector4<uchar>;
using Vector4d = Vector4<double>;

//! Vector Maths
//! Defined basic mathematic operations
template <class T, class U>
FUSION_HOST_AND_DEVICE inline Vector3<T> operator*(T S, Vector3<U> V)
{
    return Vector3<T>(V.x * S, V.y * S, V.z * S);
}

template <class T>
FUSION_HOST_AND_DEVICE inline Vector3<T> ToVector3(const Vector4<T> &V)
{
    return Vector3<T>(V.x, V.y, V.z);
}

template <class T>
FUSION_HOST_AND_DEVICE inline Vector3f operator*(const Vector3<T> &V, float S)
{
    return Vector3f(V.x * S, V.y * S, V.z * S);
}

template <class T>
FUSION_HOST_AND_DEVICE inline Vector3f ToVector3f(const Vector3<T> &V)
{
    return Vector3f((float)V.x, (float)V.y, (float)V.z);
}

template <class T>
FUSION_HOST_AND_DEVICE inline Vector3<T> floor(const Vector3<T> &V)
{
    return Vector3<T>(std::floor(V.x), std::floor(V.y), std::floor(V.z));
}

FUSION_HOST_AND_DEVICE inline Vector3f normalised(const Vector3f &V)
{
    return V / V.norm();
}

FUSION_HOST_AND_DEVICE inline Vector3i operator*(const Vector3i V, int S)
{
    return Vector3i(V.x * S, V.y * S, V.z * S);
}

FUSION_HOST_AND_DEVICE inline Vector3f operator+(Vector3i V1, Vector3f V2)
{
    return Vector3f(V1.x + V2.x, V1.y + V2.y, V1.z + V2.z);
}

FUSION_HOST_AND_DEVICE inline Vector3c ToVector3c(const Vector3f &V)
{
    return Vector3c((int)V.x, (int)V.y, (int)V.z);
}

FUSION_HOST_AND_DEVICE inline Vector3i ToVector3i(const Vector3f &V)
{
    Vector3i b((int)V.x, (int)V.y, (int)V.z);
    b.x = b.x > V.x ? b.x - 1 : b.x;
    b.y = b.y > V.y ? b.y - 1 : b.y;
    b.z = b.z > V.z ? b.z - 1 : b.z;
    return b;
}

} // namespace fusion

#endif
