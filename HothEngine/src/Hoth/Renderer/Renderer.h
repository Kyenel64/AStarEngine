#pragma once
#define STB_IMAGE_IMPLEMENTATION

#include "macros.h"
#include "Shader.h"
#include "RT.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <string>

class HOTH_API Renderer
{
public:
	Renderer(int SCR_WIDTH, int SCR_HEIGHT, std::string title, int GLVERSION_MAJOR, int GLVERSION_MINOR);
	~Renderer();

	void Render(RayTracer* RT);
	void Terminate();

private:
	GLFWwindow* window;
	int width, height;
};