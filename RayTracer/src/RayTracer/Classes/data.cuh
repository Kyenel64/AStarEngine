#pragma once
#include "vec3.cuh"

enum RenderMode
{
	Solid,
	Render
};

enum Direction
{
	FORWARD,
	BACKWARD,
	LEFT,
	RIGHT,
	UP,
	DOWN
};

enum materialType
{
	lambertian,
	metal,
	dielectric
};



struct objectData
{
	int id = 0;
	vec3 Pos = vec3(0, 0, 0);
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
	float fov = 20.0; //
	float focal_length = 1.0;
	point3 origin = point3(0, 1, 5);
	point3 lookAt = point3(0, 0, 0);
	vec3 up = vec3(0, 1, 0);
	float dist_to_focus = 5.0;
	float aperture = 0.00001;

	// Object properties
	int objectCount = 1;
	objectData defaultSphere;
	objectData objData[1000] = { defaultSphere };

	// Material properties
	int materialCount = 1;
	materialData defaultMat;
	materialData matData[100] = { defaultMat };
};