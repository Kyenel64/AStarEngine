#pragma once

#include "ray.cuh"
#include "macros.cuh"

struct RT_API hit_record
{
    point3 p; // 3D point of hit
    vec3 normal; // normal
    double t; // t value in ray equation
    bool front_face;

    __device__ inline void set_face_normal(const Ray& r, const vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class RT_API Hittable
{
public:
    __device__ virtual bool hit(const Ray& r, double t_min, double t_max, hit_record& rec) const = 0;
};