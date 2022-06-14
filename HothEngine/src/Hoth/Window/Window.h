#pragma once

#include "macros.h"

#include <GLFW/glfw3.h>

#include <string>

class HOTH_API Window
{
public:
	Window(int SCR_WIDTH, int SCR_HEIGHT, std::string title, int GLVERSION_MAJOR, int GLVERSION_MINOR);
	~Window();

	void Start();

private:
	GLFWwindow* window;
};