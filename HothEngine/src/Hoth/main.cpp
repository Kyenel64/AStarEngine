#include <Hoth/Renderer/Renderer.h>
#include "json.hpp"
#include <fstream>
#include <ostream>
#include <vector>
using namespace nlohmann;

void deserialize(Data* data, json& j);
vec3 toVec3(std::vector<float> vec);

int main()
{
	// load save file
	std::ifstream saveFile("../save/save.hoth");
	json j;
	saveFile >> j;
	Data* data = new Data;
	deserialize(data, j);

	RayTracer* RT = new RayTracer(data);

	Renderer r(RT, data, data->title, 4, 3);

	while (true)
	{
		r.processInput();
		r.Render();
	}

	return 0;
}

void deserialize(Data* data, json& j)
{
	data->title = j["title"];
	data->aspect_ratio = j["aspect_ratio"];
	data->image_width = j["image_width"];
	data->image_height = j["image_height"];
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
}

vec3 toVec3(std::vector<float> vec)
{
	return vec3(vec[0], vec[1], vec[2]);
}