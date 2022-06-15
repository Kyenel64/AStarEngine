#include "RayTracer.h"

#include <iostream>

RayTracer::RayTracer()
{
	frame = new unsigned char[1920 * 1080 * 3 * sizeof(float)];
}

RayTracer::~RayTracer()
{
	delete[] frame;
}

// Returns frame to render to texture
unsigned char* RayTracer::getFrame() const
{
	return frame;
}

// fills frame[] with render
bool RayTracer::GenerateFrame(double time)
{
	for (int i = 0; i < 1080; i++)
	{
		for (int j = 0; j < 1920; j++)
		{
			int index = i * 1920 * 3 + j * 3;
			frame[index + 0] = unsigned char(40 * time);
			frame[index + 1] = unsigned char(135 * time);
			frame[index + 2] = unsigned char(90 * time);
		}
	}
	return true;
}