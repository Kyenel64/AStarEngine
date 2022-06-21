#include "RayTracer.cuh"

#include "curand_kernel.h"

#include <iostream>
#include <chrono>

#ifdef __INTELLISENSE__
#define CUDA_KERNEL(...)
#else
#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#endif

__device__ void write_color(unsigned char* frame, int pixel_index, color pixel_color, int samples_per_pixel);
__device__ color ray_color(const Ray& r, Hittable** world, curandState *local_rand_state, Data* data);
__global__ void render(unsigned char* frame, Data* data, Hittable** world, Camera** camera, curandState *rand_state);
__global__ void create_world(Hittable** d_list, Hittable** d_world, Camera** d_camera, Material** d_matList, Data* data);
__global__ void free_world(Hittable** d_list, Hittable** d_world, Camera** d_camera, Material** d_matList, Data* data);
__global__ void render_init(int max_x, int max_y, curandState* rand_state);

RayTracer::RayTracer(Data* data) : data(data)
{
	int num_pixels = data->image_width * data->image_height;
	frame_size = 3 * num_pixels * sizeof(float);
	blockX = 8;
	blockY = 8;

	// ------------------ Allocations -----------------------
	cudaMalloc(&d_list, data->objectCount * sizeof(Hittable *));
	cudaMalloc(&d_world, sizeof(Hittable *));
	cudaMalloc(&d_data, sizeof(Data));
	cudaMemcpy(d_data, data, sizeof(Data), cudaMemcpyHostToDevice);
	cudaMalloc(&d_camera, sizeof(Camera));
	cudaMalloc(&d_matList, data->materialCount * sizeof(Material *));

	cudaMallocManaged(&frame, frame_size);
	cudaMallocManaged(&d_rand_state, num_pixels * sizeof(curandState));
	dim3 blocks(data->image_width / blockX + 1, data->image_height / blockY + 1);
	dim3 threads(blockX, blockY);

	// ------------------- Kernel calls ---------------------
	create_world CUDA_KERNEL(1, 1)(d_list, d_world, d_camera, d_matList, d_data);
	cudaDeviceSynchronize();
	render_init CUDA_KERNEL(blocks, threads)(data->image_width, data->image_height, d_rand_state);
	cudaDeviceSynchronize();

}

RayTracer::~RayTracer()
{
	free_world CUDA_KERNEL(1, 1)(d_list, d_world, d_camera, d_matList, d_data);
	cudaFree(frame);
	cudaFree(d_list);
	cudaFree(d_world);
	cudaFree(d_data);
	cudaFree(d_camera);
	cudaFree(d_matList);
	
}

// Returns final rendered frame
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

	render CUDA_KERNEL(blocks, threads)(frame, d_data, d_world, d_camera, d_rand_state);
	cudaDeviceSynchronize();


	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cerr << "\rFinished in: " << duration.count() / 1000.0 << "ms" << std::flush;

	return true;
}

// Write color to array
__device__ void write_color(unsigned char* frame, int pixel_index, color pixel_color, int samples_per_pixel)
{
	float r = pixel_color.r();
	float g = pixel_color.g();
	float b = pixel_color.b();

	// Divide color by number of samples. Gamma correct.
	float scale = 1.0 / samples_per_pixel;
	r = sqrtf(scale * r);
	g = sqrtf(scale * g);
	b = sqrtf(scale * b);

	frame[pixel_index + 0] = int(256 * clamp(r, 0.0, 0.999));
	frame[pixel_index + 1] = int(256 * clamp(g, 0.0, 0.999));
	frame[pixel_index + 2] = int(256 * clamp(b, 0.0, 0.999));
}

// Return color of pixel
__device__ color ray_color(const Ray& r, Hittable **world, curandState *local_rand_state, Data* data)
{
	Ray cur_ray = r;
	vec3 cur_attenuation = vec3(1, 1, 1);
	for (int i = 0; i < data->max_depth; i++)
	{
		hit_record rec;
		if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec))
		{
			Ray scattered;
			vec3 atteuation;
			if (rec.mat_ptr->scatter(cur_ray, rec, atteuation, scattered, local_rand_state))
			{
				cur_attenuation *= atteuation;
				cur_ray = scattered;
			}
			else
			{
				return vec3(0, 0, 0);
			}
		}
		else
		{
			vec3 unit_direction = unit_vector(cur_ray.direction());
			float t = 0.5f * (unit_direction.y() + 1.0f);
			vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
			return cur_attenuation * c;
		}
	}
	return vec3(0.0, 0.0, 0.0); // exceeded recursion
}

__global__ void render(unsigned char* frame, Data* data, Hittable **world, Camera **camera, curandState *rand_state) {
	// Initializations
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= data->image_width) || (j >= data->image_height))
		return;
	int pixel_index = j * data->image_width * 3 + i * 3;
	int rand_index = j * data->image_width + i;
	curandState local_rand_state = rand_state[rand_index];

	color pixel_color;
	for (int s = 0; s < data->samples_per_pixel; s++)
	{
		float u = float(i + curand_uniform(&local_rand_state)) / float(data->image_width - 1);
		float v = float(j + curand_uniform(&local_rand_state)) / float(data->image_height - 1);
		Ray r = (*camera)->get_ray(u, v);
		pixel_color += ray_color(r, world, &local_rand_state, data);
	}
	write_color(frame, pixel_index, pixel_color, data->samples_per_pixel);
}

__global__ void render_init(int max_x, int max_y, curandState* rand_state)
{
	// x index and y index
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y))
		return;
	int pixel_index = j * max_x + i;

	// Retrieve a random value for each thread
	curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

// Allocate world
__global__ void create_world(Hittable** d_list, Hittable** d_world, Camera** d_camera, Material** d_matList, Data* data)
{
	// Allocate new objects and world
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		for (int i = 0; i < data->materialCount; i++)
		{
			if (data->matData[i].matType == 0) // lambert
				d_matList[i] = new Lambertian(data->matData[i].Col, data->matData[i].id);
			if (data->matData[i].matType == 1) // metal
				d_matList[i] = new Lambertian(data->matData[i].Col, data->matData[i].id);
			if (data->matData[i].matType == 2) // dielectric
				d_matList[i] = new Lambertian(data->matData[i].Col, data->matData[i].id);
		}

		for (int i = 0; i < data->objectCount; i++)
		{
			d_list[i] = new Sphere(data->objData[i].Pos, data->objData[i].radius, data->objData[i].id, data->objData[i].matID, d_matList[data->objData[i].matID]);
		}
		*d_world = new Hittable_list(d_list, data->objectCount);
		*d_camera = new Camera();
	}
}

// Deallocate world
__global__ void free_world(Hittable** d_list, Hittable** d_world, Camera** d_camera, Material** d_matList, Data* data)
{
	for (int i = 0; i < data->objectCount; i++)
	{
		delete d_list[i];
	}

	for (int i = 0; i < data->materialCount; i++)
	{
		delete d_matList[i];
	}

	//delete ((Sphere*)d_list[0])->mat_ptr;
	delete* d_world;
	delete* d_camera;
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

__global__ void saveKernel(Hittable** world, Material** matList, Data* data)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		for (int i = 0; i < data->objectCount; i++)
		{
			data->objData[i].id = (*world)->getID(i);
			data->objData[i].Pos = (*world)->getPosition(i);
			data->objData[i].radius = (*world)->getRadius(i);
			data->objData[i].matID = (*world)->getMatID(i);
		};

		for (int i = 0; i < data->materialCount; i++)
		{
			if (matList[i]->getType() == lambertian)
			{
				data->matData[i].id = matList[i]->getID();
				data->matData[i].Col = matList[i]->getCol();
				data->matData[i].matType = matList[i]->getType();
			}
			
		}
	}
	
}

void RayTracer::save()
{
	saveKernel CUDA_KERNEL(1, 1)(d_world, d_matList, d_data);
	cudaDeviceSynchronize();
	cudaMemcpy(data, d_data, sizeof(Data), cudaMemcpyDeviceToHost);
}

//__global__ void addObjectKernel(Hittable** d_list, Hittable** d_world, Material** d_matList, Data* data, vec3 Pos, float radius)
//{
//	if (threadIdx.x == 0 && blockIdx.x == 0)
//	{
//		delete* d_world;
//		*d_world = new Hittable_list(d_list, data->objectCount + 1);
//		d_matList[data->materialCount] = new Lambertian(color(1.0, 0.5, 0.5), data->materialCount);
//		d_list[data->objectCount] = new Sphere(Pos, radius, data->objectCount, data->materialCount, d_matList[data->materialCount]);
//		data->objectCount++;
//		data->materialCount++;
//	}
//}
//
//void RayTracer::addObject(int id, vec3 Pos, float radius)
//{
//	cudaFree(d_list);
//	cudaMalloc(&d_list, data->objectCount+1 * sizeof(Hittable*));
//	cudaFree(d_matList);
//	cudaMalloc(&d_matList, data->materialCount + 1 * sizeof(Material*));
//	addObjectKernel CUDA_KERNEL(1, 1)(d_list, d_world, d_matList, d_data, Pos, radius);
//	cudaDeviceSynchronize();
//	cudaMemcpy(data, d_data, sizeof(Data), cudaMemcpyDeviceToHost);
//}