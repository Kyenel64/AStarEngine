#pragma once

#include "hittable.cuh"
//#include "vec3.cuh"

class RT_API Sphere : public Hittable
{
public:
    __device__ Sphere(int id) : radius(0.0), ID(id) {}
    __device__ Sphere(point3 cen, float r, int id) : center(cen), radius(r), ID(id) {};

    __device__ virtual bool hit(const Ray& r, float t_min, float t_max, hit_record& rec) const override;

    __device__ virtual void setPosition(vec3& v) override;
    __host__ __device__ virtual vec3 getPosition(int id) const;
public:
    int ID;
    point3 center;
    float radius;
};
