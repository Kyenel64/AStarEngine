#pragma once

#include "Classes/macros.cuh"
#include "Classes/misc.cuh"
#include "Classes/hittable_list.cuh"
#include "Classes/sphere.cuh"
#include "Classes/camera.cuh"
#include "Classes/material.cuh"
#include "Classes/data.cuh"

#include <curand_kernel.h>


class RT_API RayTracer
{
public:
	RayTracer(Data* data);
	~RayTracer();

	bool GenerateFrame();

	void test();
	void save();
	//void addObject(int id, vec3 Pos, float radius);
	
	Data* getData() const;

	unsigned char* getFrame() const;

private:
	int blockX, blockY;			// block dimensions
	unsigned char* frame;		// frame data
	size_t frame_size;			// frame size
	Data* data;					// host data
	Data* d_data;				// device data
	Hittable** d_list;			// list of objects
	Hittable** d_world;			// pointer to scene
	Material** d_matList;
	curandState* d_rand_state;  // device random variable
	Camera** d_camera;			// camera 
	
};