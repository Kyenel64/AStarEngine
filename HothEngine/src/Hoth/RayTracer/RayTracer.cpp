#include "RayTracer.h"

#include <ctime>
#include <iostream>

RayTracer::RayTracer()
{
	
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
bool RayTracer::GenerateFrame(float time)
{
	frame = new unsigned char[1920 * 1080 * 3 * sizeof(float)];
	for (int i = 0; i < 1080; i++)
	{
		for (int j = 0; j < 1920; j++)
		{
			int index = i * 1920 * 3 + j * 3;
			frame[index + 0] = 40 * time;
			frame[index + 1] = 135 * time;
			frame[index + 2] = 90 * time;
		}
	}
	return true;
}