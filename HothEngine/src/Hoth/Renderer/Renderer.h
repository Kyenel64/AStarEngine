#pragma once

#include "Hoth/macros.h"
#include "Shader.h"

#include "RT.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <string>
class Renderer
{
public:
	Renderer(Data& data, std::string title, int GLVERSION_MAJOR, int GLVERSION_MINOR);
	~Renderer();

	void Render(RayTracer* RT);
	void Terminate();

private:
	GLFWwindow* window;
	Data data;
};