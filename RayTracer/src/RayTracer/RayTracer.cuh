#pragma once

#include "macros.cuh"


struct RT_API Data // serialize
{
	int image_width = 1920;
	int image_height = 1080;
};


class RT_API RayTracer
{
public:
	RayTracer(Data &data);
	~RayTracer();

	bool GenerateFrame(double time);

	unsigned char* getFrame() const;
private:
	int width, height;		// frame dimensions
	int blockX, blockY;		// block dimensions
	unsigned char* frame;	// frame data
	size_t frame_size;		// frame size
	Data data;
	
};