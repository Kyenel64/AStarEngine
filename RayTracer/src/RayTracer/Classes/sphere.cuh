#pragma once

#include "hittable.cuh"
//#include "vec3.cuh"

class RT_API Sphere : public Hittable
{
public:
    __device__ Sphere() : radius(0.0) {}
    __device__ Sphere(point3 cen, float r) : center(cen), radius(r) {};

    __device__ virtual bool hit(const Ray& r, float t_min, float t_max, hit_record& rec) const override;

public:
    point3 center;
    float radius;
};
