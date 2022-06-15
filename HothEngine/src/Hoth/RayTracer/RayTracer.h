#pragma once

class RayTracer
{
public:
	RayTracer();
	~RayTracer();
	bool GenerateFrame(float time);

	unsigned char* getFrame() const;
private:
	unsigned char* frame;
};