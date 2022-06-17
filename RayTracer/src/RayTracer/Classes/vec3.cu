#include "vec3.cuh"

__host__ __device__ double vec3::length() const
{
    return sqrt(length_squared());
}

__host__ __device__ double vec3::length_squared() const
{
    return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
}