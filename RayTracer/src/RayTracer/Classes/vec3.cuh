#pragma once

#include <cmath>
#include <iostream>

#include "macros.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


using std::sqrt;

class RT_API vec3 {
public:
    __host__ __device__ vec3() : e{ 0,0,0 } {}
    __host__ __device__ vec3(double e0, double e1, double e2) : e{ e0, e1, e2 } {}

    __host__ __device__ inline double x() const { return e[0]; }
    __host__ __device__ inline double y() const { return e[1]; }
    __host__ __device__ inline double z() const { return e[2]; }
    __host__ __device__ inline double r() const { return e[0]; }
    __host__ __device__ inline double g() const { return e[1]; }
    __host__ __device__ inline double b() const { return e[2]; }

    __host__ __device__ inline const vec3& operator+() const { return *this; }
    __host__ __device__ inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    __host__ __device__ inline double operator[](int i) const { return e[i]; }
    __host__ __device__ inline double& operator[](int i) { return e[i]; }

    __host__ __device__ double length() const;
    __host__ __device__ double length_squared() const;
public:
    double e[3];
};

using point3 = vec3;
using color = vec3;

__host__ __device__ inline std::ostream& operator<<(std::ostream& out, const vec3& v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

__host__ __device__ inline vec3 operator+(const vec3& u, const vec3& v) {
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3& u, const vec3& v) {
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& u, const vec3& v) {
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__ inline vec3 operator*(double t, const vec3& v) {
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& v, double t) {
    return t * v;
}

__host__ __device__ inline vec3 operator/(vec3 v, double t)
{
    return (1 / t) * v;
}

__host__ __device__ inline double dot(const vec3& u, const vec3& v)
{
    return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

__host__ __device__ inline vec3 cross(const vec3& u, const vec3& v)
{
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
        u.e[2] * v.e[0] - u.e[0] * v.e[2],
        u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__host__ __device__ inline vec3 unit_vector(vec3 v)
{
    return v / v.length();
}
