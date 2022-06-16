#pragma once

#include "vec3.cuh"

class Ray {
public:
    __device__ Ray() {}
    __device__ Ray(const point3& origin, const vec3& direction) : orig(origin), dir(direction) {}

    __device__ point3 origin() const { return orig; }
    __device__ vec3 direction() const { return dir; }

    __device__ point3 at(float t) const {
        return orig + t * dir;
    }

public:
    point3 orig;
    vec3 dir;
};