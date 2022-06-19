#pragma once

#include "Classes/macros.cuh"
#include "Classes/misc.cuh"
#include "Classes/hittable_list.cuh"
#include "Classes/sphere.cuh"

struct objectData
{
	int id = 0;
	vec3 Pos = vec3(0, 0, -3);
	float radius = 1;
};

struct Data
{
	std::string title = "untitled";
	float aspect_ratio = float(16.0 / 9.0);
	int image_width = 1920;
	int image_height = 1080;

	// Camera properties
	float viewport_height = 2.0;
	float viewport_width = aspect_ratio * viewport_height;
	float focal_length = 1.0;
	point3 origin = vec3(0, 0, 0);
	vec3 horizontal = vec3(viewport_width, 0, 0);
	vec3 vertical = vec3(0, viewport_height, 0);
	point3 lower_left_corner = origin - horizontal / 2 - vertical / 2 - vec3(0, 0, focal_length);

	int objectCount = 1;
	objectData defaultSphere;
	objectData objData[1000] = {defaultSphere};
};


class RT_API RayTracer
{
public:
	RayTracer(Data* data);
	~RayTracer();

	bool GenerateFrame();

	void test();
	void save();
	void addObject(int id, vec3 Pos, float radius);
	
	Data* getData() const;

	unsigned char* getFrame() const;

private:
	int blockX, blockY;		// block dimensions
	unsigned char* frame;	// frame data
	size_t frame_size;		// frame size
	Data* data;
	Data* d_data;
	Hittable **d_list;
	Hittable **d_world;
	
};