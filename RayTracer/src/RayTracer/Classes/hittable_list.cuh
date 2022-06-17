#pragma once

#include "hittable.cuh"

#include <memory>
#include <vector>

class RT_API Hittable_list : public Hittable {
public:
    __device__ Hittable_list() : list(nullptr), list_size(0) {}

    __device__ Hittable_list(Hittable** object, int n)
    {
        list = object;
        list_size = n;
    }

    __device__ virtual bool hit(const Ray& r, float t_min, float t_max, hit_record& rec) const override;

public:
    Hittable** list;
    int list_size;
};