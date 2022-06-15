#pragma once

#include "macros.h"

class HOTH_API RayTracer
{
public:
	RayTracer();
	~RayTracer();
	bool GenerateFrame(double time);

	unsigned char* getFrame() const;
private:
	unsigned char* frame;
};