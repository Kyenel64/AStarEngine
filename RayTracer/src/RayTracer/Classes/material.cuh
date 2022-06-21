#ifndef MATERIAL_CUH
#define MATERIAL_CUH

struct hit_record;

#include "ray.cuh"
#include "hittable.cuh"
#include "data.cuh"


class Material
{
public:
    __device__ virtual bool scatter(const Ray& r_in, const hit_record& rec, color& attenuation, Ray& scattered, curandState* local_rand_state) const = 0;

    __device__ virtual int getID() const = 0;
    __device__ virtual color getCol() const = 0;
    __device__ virtual materialType getType() const = 0;
};

// Diffuse
class Lambertian : public Material
{
public:
    __device__ Lambertian(const color& a, int id) : albedo(a), ID(id) {}

    __device__ virtual bool scatter(const Ray& r_in, const hit_record& rec, color& attenuation, Ray& scattered, curandState* local_rand_state) const override
    {
        vec3 scatter_direction = rec.normal + random_unit_vector(local_rand_state);
        scattered = Ray(rec.p, scatter_direction);
        attenuation = albedo;
        return true;
    }

    __device__ virtual int getID() const { return ID; }
    __device__ virtual color getCol() const { return albedo; }
    __device__ virtual materialType getType() const { return type; }

public:
    int ID;
    color albedo;
    materialType type = lambertian;
};

#endif