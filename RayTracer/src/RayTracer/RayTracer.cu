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
__global__ void render(unsigned char* frame, Data d, Hittable** world);
__global__ void create_world(Hittable** d_list, Hittable** d_world);
__global__ void free_world(Hittable** d_list, Hittable** d_world);

RayTracer::RayTracer(Data &data) : data(data)
{
	int num_pixels = data.image_width * data.image_height;
	frame_size = 3 * num_pixels * sizeof(double);
	blockX = 4;
	blockY = 4;

	// -------------------- World -----------------------
	cudaMallocManaged(&d_list, 2 * sizeof(Hittable *));
	cudaMallocManaged(&d_world, sizeof(Hittable *));
	create_world CUDA_KERNEL(1, 1)(d_list, d_world);
	cudaDeviceSynchronize();

	// ------------------- Memory -----------------------
	cudaMallocManaged(&frame, frame_size);

}

RayTracer::~RayTracer()
{
	free_world CUDA_KERNEL(1, 1)(d_list, d_world);
	cudaFree(frame);
	cudaFree(d_list);
	cudaFree(d_world);
	
}

// Returns frame to render to texture
unsigned char* RayTracer::getFrame() const
{
	return frame;
}

// fills frame[] with render. Acts like main()
bool RayTracer::GenerateFrame()
{
	auto start = std::chrono::high_resolution_clock::now();


	dim3 blocks(data.image_width / blockX + 1, data.image_height / blockY + 1);
	dim3 threads(blockX, blockY);

	render CUDA_KERNEL(blocks, threads)(frame, data, d_world);
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
		printf("test");
		return 0.5 * (rec.normal + color(1, 1, 1));
	}

	// background color
	vec3 unit_direction = unit_vector(r.direction());
	auto t = 0.5 * (unit_direction.y() + 1.0);
	return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}

__global__ void render(unsigned char* frame, Data d, Hittable **world) {
	// initialize variables and random state
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= d.image_width) || (j >= d.image_height))
		return;
	int pixel_index = j * d.image_width * 3 + i * 3;

	double u = double(i) / double(d.image_width - 1);
	double v = double(j) / double(d.image_height - 1);

	Ray r(d.origin, d.lower_left_corner + u * d.horizontal + v * d.vertical);
	write_color(frame, pixel_index, ray_color(r, world));
}

// Allocate world
__global__ void create_world(Hittable** d_list, Hittable** d_world)
{
	// Allocate new objects and world
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		*d_list = new Sphere(vec3(0, 0, -1), 0.5);
		*(d_list + 1) = new Sphere(vec3(0, -100.5, -1), 100);
		*d_world = new Hittable_list(d_list, 2);
	}
}

// Deallocate world
__global__ void free_world(Hittable** d_list, Hittable** d_world)
{
	delete* (d_list);
	delete* (d_list + 1);
	delete* d_world;
}

