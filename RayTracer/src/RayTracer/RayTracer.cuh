#pragma once

#include "Classes/macros.cuh"
#include "Classes/misc.cuh"
#include "Classes/hittable_list.cuh"
#include "Classes/sphere.cuh"


struct Data // serialize
{
	std::string title;
	float aspect_ratio;
	int image_width;
	int image_height;

	// Camera properties
	float viewport_height;
	float viewport_width;
	float focal_length;
	point3 origin;
	vec3 horizontal;
	vec3 vertical;
	point3 lower_left_corner;

	int objectCount;
	vec3 spherePos1;
};


class RT_API RayTracer
{
public:
	RayTracer(Data &data);
	~RayTracer();

	bool GenerateFrame();

	void test();
	void save();

	unsigned char* getFrame() const;

private:
	int blockX, blockY;		// block dimensions
	unsigned char* frame;	// frame data
	size_t frame_size;		// frame size
	Data& data;
	Hittable **d_list;
	Hittable **d_world;
	
};