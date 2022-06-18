#include <Hoth/Renderer/Renderer.h>
#include "json.hpp"
#include <fstream>
#include <ostream>
#include <vector>
using namespace nlohmann;

void to_data(Data& data, json& j);
vec3 toVec3(std::vector<float> vec);

int main()
{
	std::ifstream saveFile("../save/save.json");
	json j;
	saveFile >> j;

	Data data;
	to_data(data, j);

	RayTracer* RT = new RayTracer(data);

	Renderer r(RT, data, data.title, 4, 3);

	while (true)
	{
		r.processInput();
		r.Render();
	}

	return 0;
}

void to_data(Data& data, json& j)
{
	data.title = j["title"];
	data.aspect_ratio = j["aspect_ratio"];
	data.image_width = j["image_width"];
	data.image_height = j["image_height"];
	data.viewport_height = j["viewport_height"];
	data.viewport_width = j["viewport_width"];
	data.focal_length = j["focal_length"];
	data.origin = toVec3(j["origin"]);
	data.horizontal = toVec3(j["horizontal"]);
	data.vertical = toVec3(j["vertical"]);
	data.lower_left_corner = toVec3(j["lower_left_corner"]);
	
	data.objectCount = j["objectCount"];
	data.spherePos1 = toVec3(j["spherePos1"]);
}

vec3 toVec3(std::vector<float> vec)
{
	return vec3(vec[0], vec[1], vec[2]);
}