#pragma once
#define STB_IMAGE_IMPLEMENTATION

#include "macros.h"
#include "Shader.h"
#include "RayTracer/RayTracer.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "stb_image.h"

#include <string>

class HOTH_API Renderer
{
public:
	Renderer(int SCR_WIDTH, int SCR_HEIGHT, std::string title, int GLVERSION_MAJOR, int GLVERSION_MINOR);
	~Renderer();

	void Start();

private:
	GLFWwindow* window;
	unsigned char* data;
	Shader* ourShader;
	unsigned int VBO, VAO, EBO;
	int width, height;
};