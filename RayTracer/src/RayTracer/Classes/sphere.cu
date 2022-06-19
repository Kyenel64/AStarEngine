#include "sphere.cuh"

__device__ bool Sphere::hit(const Ray& r, float t_min, float t_max, hit_record& rec) const
{
    vec3 oc = r.origin() - center;
    float a = r.direction().length_squared();
    float b = dot(oc, r.direction());
    float c = oc.length_squared() - radius * radius;
    float discriminant = b * b - a * c;
    if (discriminant < 0.0)
        return false;
    float sqrtd = sqrtf(discriminant);

    float root = (-b - sqrtd) / a;
    if (root < t_min || t_max < root)
    {
        root = (-b + sqrtd) / a;
        if (root < t_min || t_max < root)
            return false;
    }

    // Fill hit record
    rec.t = root;
    rec.p = r.at(rec.t);
    vec3 outward_normal = (rec.p - center) / radius;
    rec.set_face_normal(r, outward_normal);

    return true;
}

__device__ void Sphere::setPosition(vec3 v)
{
    center = v;
}

__host__ __device__ vec3 Sphere::getPosition(int id) const
{
    return center;
}

__host__ __device__ float Sphere::getRadius(int id) const
{
    return radius;
}

__host__ __device__ int Sphere::getID(int id) const
{
    return ID;
}