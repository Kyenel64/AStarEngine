#include "RayTracer.cuh"
#include "Classes/vec3.cuh"
#include "Classes/ray.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

#ifdef __INTELLISENSE__
#define CUDA_KERNEL(...)
#else
#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#endif

RayTracer::RayTracer(int width, int height) : width(width), height(height)
{
	int num_pixels = width * height;
	frame_size = 3 * num_pixels * sizeof(float);
	cudaMallocManaged(&frame, frame_size);
	blockX = 4;
	blockY = 4;

}

RayTracer::~RayTracer()
{
	cudaFree(frame);
}

// Returns frame to render to texture
unsigned char* RayTracer::getFrame() const
{
	return frame;
}

__device__ void write_color(unsigned char* frame, int pixel_index, color pixel_color)
{
	frame[pixel_index + 0] = int(255.99 * (pixel_color.r()));
	frame[pixel_index + 1] = int(255.99 * (pixel_color.g()));
	frame[pixel_index + 2] = int(255.99 * (pixel_color.b()));
}

__device__ color ray_color(const Ray& r)
{
	vec3 unit_direction = unit_vector(r.direction());
	float t = 0.5 * (unit_direction.y() + 1.0);
	return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}

__global__ void render(unsigned char* frame, int max_x, int max_y) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x * 3 + i * 3;

	color pixel_color(float(i) / max_x, float(j) / max_y, 0.2);
	write_color(frame, pixel_index, pixel_color);
}

// fills frame[] with render. Acts like main()
bool RayTracer::GenerateFrame(double time)
{
	dim3 blocks(width / blockX + 1, height / blockY + 1);
	dim3 threads(blockX, blockY);
	render CUDA_KERNEL(blocks, threads)(frame, width, height);
	cudaDeviceSynchronize();


	return true;
}

