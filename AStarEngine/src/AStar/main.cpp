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

	data->title = j["title"];
	data->aspect_ratio = j["aspect_ratio"];
	data->image_width = j["image_width"];
	data->image_height = j["image_height"];

	data->samples_per_pixel = j["samples_per_pixel"];

	data->viewport_height = j["viewport_height"];
	data->viewport_width = j["viewport_width"];
	data->focal_length = j["focal_length"];
	data->origin = toVec3(j["origin"]);
	data->horizontal = toVec3(j["horizontal"]);
	data->vertical = toVec3(j["vertical"]);
	data->lower_left_corner = toVec3(j["lower_left_corner"]);
	
	data->objectCount = j["objectCount"];
	
	for (int i = 0; i < data->objectCount; i++)
	{
		data->objData[i].id = j["objectData"][i][0];
		data->objData[i].Pos = toVec3(j["objectData"][i][1]);
		data->objData[i].radius = j["objectData"][i][2];
	}

	return true;
}

vec3 toVec3(std::vector<float> vec)
{
	return vec3(vec[0], vec[1], vec[2]);
}