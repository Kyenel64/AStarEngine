#ifndef CAMERA_CUH
#define CAMERA_CUH

#include "misc.cuh"
#include "data.cuh"


class RT_API Camera
{
public:
    __device__ Camera(Data* data) {
        float theta = degrees_to_radians(data->fov);
        float h = tanf(theta / 2);
        float viewport_height = float(2.0 * h);
        float viewport_width = data->aspect_ratio * viewport_height;

        w = unit_vector(data->origin - data->lookAt);
        u = unit_vector(cross(data->up, w));
        v = cross(w, u);

        origin = data->origin;
        horizontal = data->dist_to_focus * viewport_width * u;
        vertical = data->dist_to_focus * viewport_height * v;
        lower_left_corner = origin - horizontal / 2 - vertical / 2 - data->dist_to_focus * w;

        lens_radius = data->aperture / 2;
    }

    __device__ Ray get_ray(float s, float t, curandState* local_rand_state) const
    {
        vec3 rd = lens_radius * random_in_unit_disk(local_rand_state);
        vec3 offset = u * rd.x() + v * rd.y();
        return Ray(origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset);
    }

private:
    point3 origin;
    point3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
    float lens_radius;
};

#endif