#ifndef MATERIAL_CUH
#define MATERIAL_CUH

struct hit_record;

#include "ray.cuh"
#include "hittable.cuh"


class Material
{
public:
    __device__ virtual bool scatter(const Ray& r_in, const hit_record& rec, color& attenuation, Ray& scattered, curandState* local_rand_state) const = 0;
};

// Diffuse
class lambertian : public Material
{
public:
    __device__ lambertian(const color& a) : albedo(a) {}

    __device__ virtual bool scatter(const Ray& r_in, const hit_record& rec, color& attenuation, Ray& scattered, curandState* local_rand_state) const override
    {
        vec3 scatter_direction = rec.normal + random_unit_vector(local_rand_state);
        scattered = Ray(rec.p, scatter_direction);
        attenuation = albedo;
        return true;
    }

public:
    color albedo;
};

#endif