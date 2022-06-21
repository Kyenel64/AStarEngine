#pragma once

#include "ray.cuh"
#include "macros.cuh"

class Material;

struct RT_API hit_record
{
    point3 p; // 3D point of hit
    vec3 normal; // normal
    float t; // t value in ray equation
    Material* mat_ptr;
    bool front_face;

    __device__ inline void set_face_normal(const Ray& r, const vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class RT_API Hittable
{
public:
    __device__ virtual bool hit(const Ray& r, float t_min, float t_max, hit_record& rec) const = 0;

    __device__ virtual void setPosition(vec3 v) = 0;

    __host__ __device__ virtual vec3 getPosition(int id) const = 0;
    __host__ __device__ virtual float getRadius(int id) const = 0;
    __host__ __device__ virtual int getID(int id) const = 0;
    __host__ __device__ virtual int getMatID(int id) const = 0;
};
