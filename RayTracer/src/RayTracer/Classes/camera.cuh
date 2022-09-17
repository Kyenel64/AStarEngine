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
        viewport_height = float(2.0 * h);
        viewport_width = data->aspect_ratio * viewport_height;
        dist_to_focus = data->dist_to_focus;
        up = data->up;

        Front = unit_vector(data->origin - data->lookAt);
        Right = unit_vector(cross(data->up, Front));
        Up = cross(Front, Right);

        yaw = 90.0;
        pitch = 0.0;

        lookAt = data->lookAt;
        origin = data->origin;
        horizontal = data->dist_to_focus * viewport_width * Right;
        vertical = data->dist_to_focus * viewport_height * Up;
        lower_left_corner = origin - horizontal / 2 - vertical / 2 - data->dist_to_focus * Front;

        lens_radius = data->aperture / 2;
    }

    __device__ Ray get_ray(float s, float t, curandState* local_rand_state) const
    {
        vec3 rd = lens_radius * random_in_unit_disk(local_rand_state);
        vec3 offset = Right * rd.x() + Up * rd.y();
        return Ray(origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset);
    }

    __device__ vec3 getDirection() const
    {
        return lookAt;
    }

    __device__ RenderMode getRenderMode() const
    {
        return renderMode;
    }

    __device__ void setRenderMode(RenderMode mode)
    {
        renderMode = mode;
    }

    __device__ void move(Direction dir, float deltaTime)
    {
        float velocity = movementSpeed * deltaTime;
        if (dir == FORWARD)
        {
            origin -= Front * velocity;
        }
        if (dir == BACKWARD)
        {
            origin += Front * velocity;
        }
        if (dir == LEFT)
        {
            origin -= Right * velocity;
        }
        if (dir == RIGHT)
        {
            origin += Right * velocity;
        }

        //Front = unit_vector(origin - lookAt);
        Front = vec3(cosf(degrees_to_radians(yaw)) * cosf(degrees_to_radians(pitch)),
            sinf(degrees_to_radians(pitch)), sinf(degrees_to_radians(yaw) * cos(degrees_to_radians(pitch))));
        Front = unit_vector(Front);
        Right = unit_vector(cross(up, Front));
        Up = cross(Front, Right); 

        horizontal = dist_to_focus * viewport_width * Right;
        vertical = dist_to_focus * viewport_height * Up;
        lower_left_corner = origin - horizontal / 2 - vertical / 2 - dist_to_focus * Front;
    }

    __device__ void processMouseMove(float xoffset, float yoffset)
    {
        xoffset *= mouseSensitivity;
        yoffset *= mouseSensitivity;

        yaw += xoffset;
        pitch -= yoffset;

        Front = vec3(cosf(degrees_to_radians(yaw)) * cosf(degrees_to_radians(pitch)),
            sinf(degrees_to_radians(pitch)), sinf(degrees_to_radians(yaw) * cos(degrees_to_radians(pitch))));
        Front = unit_vector(Front);
        Right = unit_vector(cross(up, Front));
        Up = cross(Front, Right);

        horizontal = dist_to_focus * viewport_width * Right;
        vertical = dist_to_focus * viewport_height * Up;
        lower_left_corner = origin - horizontal / 2 - vertical / 2 - dist_to_focus * Front;
    }

private:
    point3 origin;
    point3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 Front, Right, Up;
    vec3 lookAt;
    vec3 up;
    float viewport_height;
    float viewport_width;

    float lens_radius;
    RenderMode renderMode = Solid;
    float movementSpeed = 5;
    float dist_to_focus;
    float mouseSensitivity = 0.2;

public:
    float yaw;
    float pitch;
};

#endif