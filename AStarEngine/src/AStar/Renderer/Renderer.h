#pragma once

#include "AStar/macros.h"
#include "Shader.h"

#include "RT.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <ostream>
#include "json.hpp"
using namespace nlohmann;

class Renderer
{
public:
	Renderer(RayTracer* RT, Data* data, std::string title, int GLVERSION_MAJOR, int GLVERSION_MINOR);
	~Renderer();

	void Render();

	void serialize();
	void processInput();

private:
	GLFWwindow* window;
	Data* data;
	RayTracer* RT;
};