#include <AStar/Renderer/Renderer.h>
#include "json.hpp"
using namespace nlohmann;

bool deserialize(Data* data, std::string path);
vec3 toVec3(std::vector<float> vec);

int main()
{
	// load save file
	Data* data = new Data;
	deserialize(data, "../save/save.astar");

	RayTracer* RT = new RayTracer(data);

	Renderer r(RT, data, data->title, 4, 3);

	std::string s;

	while (true)
	{
		r.processInput();
		r.Render();
	}

	delete data;
	delete RT;

	return 0;
}

bool deserialize(Data* data, std::string path)
{
	std::ifstream saveFile(path);
	json j;
	if (saveFile.fail())
		return false;
	saveFile >> j;

	// Image properties
	data->title = j["title"];
	data->aspect_ratio = j["aspect_ratio"];
	data->image_width = j["image_width"];
	data->image_height = j["image_height"];

	// Rendering properties
	data->samples_per_pixel = j["samples_per_pixel"];
	data->max_depth = j["max_depth"];

	// Camera properties
	data->fov = j["fov"];
	data->focal_length = j["focal_length"];
	data->origin = toVec3(j["origin"]);
	data->lookAt = toVec3(j["lookAt"]);
	data->up = toVec3(j["up"]);
	data->dist_to_focus = j["dist_to_focus"];
	data->aperture = j["aperture"];
	
	// Object properties
	data->objectCount = j["objectCount"];
	for (int i = 0; i < data->objectCount; i++)
	{
		data->objData[i].id = j["objectData"][i][0];
		data->objData[i].Pos = toVec3(j["objectData"][i][1]);
		data->objData[i].radius = j["objectData"][i][2];
		data->objData[i].matID = j["objectData"][i][3];
	}

	// Material properties
	data->materialCount = j["materialCount"];
	for (int i = 0; i < data->materialCount; i++)
	{
		data->matData[i].id = j["materialData"][i][0];
		data->matData[i].Col = toVec3(j["materialData"][i][1]);
		data->matData[i].matType = j["materialData"][i][2];
	}

	return true;
}

vec3 toVec3(std::vector<float> vec)
{
	return vec3(vec[0], vec[1], vec[2]);
}