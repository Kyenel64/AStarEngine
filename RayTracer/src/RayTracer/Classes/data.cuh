#pragma once
#include "vec3.cuh"

enum materialType
{
	lambertian,
	metal,
	dielectric
};

struct objectData
{
	int id = 0;
	vec3 Pos = vec3(0, 0, -1);
	float radius = 0.5;
	int matID = 0;
};

struct materialData
{
	int id = 0;
	color Col = color(0.5, 0.5, 0.5);
	materialType matType = lambertian;
};

struct Data
{
	// Image properties
	std::string title = "untitled";
	float aspect_ratio = float(16.0 / 9.0);
	int image_width = 1920;
	int image_height = 1080;

	// Rendering properties
	int samples_per_pixel = 1;
	int max_depth = 5;

	// Camera properties
	float viewport_height = 2.0;
	float viewport_width = aspect_ratio * viewport_height;
	float focal_length = 1.0;
	point3 origin = vec3(0, 0, 0);
	vec3 horizontal = vec3(viewport_width, 0, 0);
	vec3 vertical = vec3(0, viewport_height, 0);
	point3 lower_left_corner = origin - horizontal / 2 - vertical / 2 - vec3(0, 0, focal_length);

	int objectCount = 1;
	objectData defaultSphere;
	objectData objData[1000] = { defaultSphere };

	int materialCount = 1;
	materialData defaultMat;
	materialData matData[100] = { defaultMat };
};