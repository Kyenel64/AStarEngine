#include "hittable_list.cuh"

__device__ bool Hittable_list::hit(const Ray& r, float t_min, float t_max, hit_record& rec) const {

    hit_record temp_rec;
    bool hit_anything = false;
    auto closest_so_far = t_max;

    for (int i = 0; i < list_size; i++) {
        if (list[i]->hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}

__device__ void Hittable_list::setPosition(vec3 v)
{
    list[0]->setPosition(v);
}

__host__ __device__ vec3 Hittable_list::getPosition(int id) const
{
    return list[id]->getPosition(id);
}

__host__ __device__ float Hittable_list::getRadius(int id) const
{
    return list[id]->getRadius(id);
}

__host__ __device__ int Hittable_list::getID(int id) const
{
    return list[id]->getID(id);
}

__host__ __device__ int Hittable_list::getMatID(int id) const
{
    return list[id]->getMatID(id);
}