#pragma once

#include "macros.cuh"

class RT_API RayTracer
{
public:
	RayTracer(int width, int height);
	~RayTracer();

	bool GenerateFrame(double time);

	unsigned char* getFrame() const;
private:
	int width, height;		// frame dimensions
	int blockX, blockY;		// block dimensions
	unsigned char* frame;	// frame data
	size_t frame_size;		// frame size
};