#ifndef CAMERA_CUH
#define CAMERA_CUH

#include "misc.cuh"


class RT_API Camera
{
public:
    __device__ Camera() {
        aspect_ratio = float(16.0 / 9.0);
        viewport_height = 2.0;
        viewport_width = aspect_ratio * viewport_height;
        focal_length = 1.0;

        origin = vec3(0, 0, 0);
        horizontal = vec3(viewport_width, 0, 0);
        vertical = vec3(0, viewport_height, 0);
        lower_left_corner = origin - horizontal / 2 - vertical / 2 - vec3(0, 0, focal_length);
    }

    __device__ Ray get_ray(float u, float v) const
    {
        return Ray(origin, lower_left_corner + u * horizontal + v * vertical - origin);
    }

private:
    point3 origin;
    point3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    float aspect_ratio;
    float viewport_height;
    float viewport_width;
    float focal_length;
};

#endif