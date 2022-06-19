#include "RayTracer.cuh"

#include "curand_kernel.h"

#include <iostream>
#include <chrono>

#ifdef __INTELLISENSE__
#define CUDA_KERNEL(...)
#else
#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#endif

__device__ void write_color(unsigned char* frame, int pixel_index, color pixel_color);
__device__ color ray_color(const Ray& r, Hittable** world);
__global__ void render(unsigned char* frame, Data* data, Hittable** world);
__global__ void create_world(Hittable** d_list, Hittable** d_world, Data* data);
__global__ void free_world(Hittable** d_list, Hittable** d_world, Data* data);

RayTracer::RayTracer(Data* data) : data(data)
{
	int num_pixels = data->image_width * data->image_height;
	frame_size = 3 * num_pixels * sizeof(float);
	blockX = 8;
	blockY = 8;

	// -------------------- World -----------------------
	cudaMalloc(&d_list, data->objectCount * sizeof(Hittable *));
	cudaMalloc(&d_world, sizeof(Hittable *));
	cudaMalloc(&d_data, sizeof(Data));
	cudaMemcpy(d_data, data, sizeof(Data), cudaMemcpyHostToDevice);
	create_world CUDA_KERNEL(1, 1)(d_list, d_world, d_data);
	cudaDeviceSynchronize();

	// ------------------- Memory -----------------------
	cudaMallocManaged(&frame, frame_size);

}

RayTracer::~RayTracer()
{
	free_world CUDA_KERNEL(1, 1)(d_list, d_world, d_data);
	cudaFree(frame);
	cudaFree(d_list);
	cudaFree(d_world);
	cudaFree(d_data);
	
}

// Returns frame to render to texture
unsigned char* RayTracer::getFrame() const
{
	return frame;
}

Data* RayTracer::getData() const
{
	return data;
}

// fills frame[] with render. Acts like main()
bool RayTracer::GenerateFrame()
{
	auto start = std::chrono::high_resolution_clock::now();


	dim3 blocks(data->image_width / blockX + 1, data->image_height / blockY + 1);
	dim3 threads(blockX, blockY);


	render CUDA_KERNEL(blocks, threads)(frame, d_data, d_world);
	cudaDeviceSynchronize();


	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cerr << "\rFinished in: " << duration.count() / 1000.0 << "ms" << std::flush;

	return true;
}

// Write color to array
__device__ void write_color(unsigned char* frame, int pixel_index, color pixel_color)
{
	frame[pixel_index + 0] = int(255.99 * (pixel_color.r()));
	frame[pixel_index + 1] = int(255.99 * (pixel_color.g()));
	frame[pixel_index + 2] = int(255.99 * (pixel_color.b()));
}

// Return color of pixel
__device__ color ray_color(const Ray& r, Hittable **world)
{
	// temp hit record
	hit_record rec;
	if ((*world)->hit(r, 0, FLT_MAX, rec)) {
		return 0.5 * (rec.normal + color(1, 1, 1));
	}

	// background color
	vec3 unit_direction = unit_vector(r.direction());
	auto t = 0.5 * (unit_direction.y() + 1.0);
	return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}

__global__ void render(unsigned char* frame, Data* data, Hittable **world) {
	// initialize variables and random state
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= data->image_width) || (j >= data->image_height))
		return;
	int pixel_index = j * data->image_width * 3 + i * 3;

	float u = float(i) / float(data->image_width - 1);
	float v = float(j) / float(data->image_height - 1);

	Ray r(data->origin, data->lower_left_corner + u * data->horizontal + v * data->vertical);
	write_color(frame, pixel_index, ray_color(r, world));
}

// Allocate world
__global__ void create_world(Hittable** d_list, Hittable** d_world, Data* data)
{
	// Allocate new objects and world
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		for (int i = 0; i < data->objectCount; i++)
		{
			d_list[i] = new Sphere(data->objData[i].Pos, data->objData[i].radius, data->objData[i].id);
		}
		*d_world = new Hittable_list(d_list, data->objectCount);
	}
}

// Deallocate world
__global__ void free_world(Hittable** d_list, Hittable** d_world, Data* data)
{
	for (int i = 0; i < data->objectCount; i++)
	{
		delete d_list[i];
	}
	delete* d_world;
}

__global__ void testKernel(Hittable **world)
{
	(*world)->setPosition(vec3(1, 0, -2));
}

void RayTracer::test()
{
	testKernel CUDA_KERNEL(1, 1)(d_world);
	cudaDeviceSynchronize();
}

__global__ void saveKernel(Hittable** world, Data* data)
{
	for (int i = 0; i < data->objectCount; i++)
	{
		data->objData[i].Pos = (*world)->getPosition(i);
	}
	//data->objData[0].radius = (*world)->getRadius(0);
}

void RayTracer::save()
{
	saveKernel CUDA_KERNEL(1, 1)(d_world, d_data);
	cudaMemcpy(data, d_data, sizeof(Data), cudaMemcpyDeviceToHost);
}