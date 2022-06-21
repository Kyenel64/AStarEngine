#pragma once

#include "hittable.cuh"

class RT_API Sphere : public Hittable
{
public:
    __device__ Sphere(int id) : radius(0.0), ID(id) {}
    __device__ Sphere(point3 cen, float r, int id, int matID, Material *m) : center(cen), radius(r), ID(id), mat_ptr(m), matID(matID) {};

    __device__ virtual bool hit(const Ray& r, float t_min, float t_max, hit_record& rec) const override;

    __device__ virtual void setPosition(vec3 v) override;

    __host__ __device__ vec3 getPosition(int id) const override;
    __host__ __device__ float getRadius(int id) const override;
    __host__ __device__ int getID(int id) const override;
    __host__ __device__ int getMatID(int id) const override;
public:
    int ID, matID;
    point3 center;
    float radius;
    Material* mat_ptr;
};
