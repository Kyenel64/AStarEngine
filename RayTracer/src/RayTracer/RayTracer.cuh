#pragma once

#include "Classes/macros.cuh"
#include "Classes/misc.cuh"
#include "Classes/hittable_list.cuh"
#include "Classes/sphere.cuh"


struct RT_API Data // serialize
{
	const float aspect_ratio = float(16.0 / 9.0);
	int image_width = 1280;
	int image_height = 700;

	// Camera properties
	float viewport_height = 2.0;
	float viewport_width = aspect_ratio * viewport_height;
	float focal_length = 1.0;
	point3 origin = point3(0, 0, 0);
	vec3 horizontal = vec3(viewport_width, 0.0, 0.0);
	vec3 vertical = vec3(0.0, viewport_height, 0.0);
	point3 lower_left_corner = origin - horizontal / 2 - vertical / 2 - vec3(0, 0, focal_length);

};


class RT_API RayTracer
{
public:
	RayTracer(Data &data);
	~RayTracer();

	bool GenerateFrame(Data& data, float time);

	unsigned char* getFrame() const;

private:
	int blockX, blockY;		// block dimensions
	unsigned char* frame;	// frame data
	size_t frame_size;		// frame size
	Data data;
	Hittable **d_list;
	Hittable **d_world;
	
};