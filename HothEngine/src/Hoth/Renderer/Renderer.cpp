#include "Renderer.h"

#include <iostream>

// Simple shaders to display texture quad on screen.
const char* vertexShaderSource = 
"#version 430 core\n"
"layout(location = 0) in vec3 aPos;\n"
"layout(location = 1) in vec2 aTexCoord;\n"

"out vec3 ourColor;\n"
"out vec2 TexCoord;\n"

"void main()\n"
"{\n"
"	gl_Position = vec4(aPos, 1.0);\n"
"	TexCoord = vec2(aTexCoord.x, aTexCoord.y);\n"
"}";

const char* fragmentShaderSource =
"#version 430 core\n"
"out vec4 FragColor;\n"

"in vec3 ourColor;\n"
"in vec2 TexCoord;\n"

"uniform sampler2D texture1;\n"

"void main()\n"
"{\n"
"	FragColor = texture(texture1, TexCoord);\n"
"}";

// declarations
void framebuffer_size_callback(GLFWwindow* window, int width, int height);

Renderer::Renderer(RayTracer* RT, Data& data, std::string title, int GLVERSION_MAJOR, 
	int GLVERSION_MINOR) : RT(RT), data(data)
{
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, GLVERSION_MAJOR);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, GLVERSION_MINOR);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_MAXIMIZED, GLFW_FALSE); // Maximized window

	window = glfwCreateWindow(data.image_width, data.image_height, title.c_str(), NULL, NULL);

	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
	}

	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

	// Initialize GLAD
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
	}

	// Set viewport inside the window
	glViewport(0, 0, data.image_width, data.image_height);

	Shader* ourShader = new Shader(vertexShaderSource, fragmentShaderSource);

	float vertices[] = {
		 // positions         // texCoords
		 1.0f,  1.0f, 0.0f,   1.0f, 1.0f,
		 1.0f, -1.0f, 0.0f,   1.0f, 0.0f,
		-1.0f, -1.0f, 0.0f,   0.0f, 0.0f,
		-1.0f,  1.0f, 0.0f,   0.0f, 1.0f 
	};

	unsigned int indices[] = {
		0, 1, 3,
		1, 2, 3 
	};

	ourShader->use();

	// Generate buffers
	unsigned int VAO, VBO, EBO;
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);

	// Bind buffers
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	// position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	// texture coord attribute
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);

	// load and create a texture 
	// -------------------------
	unsigned int texture;
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

}

Renderer::~Renderer()
{
	
}

void Renderer::Render()
{
	if (glfwWindowShouldClose(window))
		glfwTerminate();

	// update Frame
	if (RT->GenerateFrame())
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, data.image_width, data.image_height, 0, GL_RGB, GL_UNSIGNED_BYTE, RT->getFrame());
	else
		std::cout << "Failed to load texture" << std::endl;


	// render frame
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);


	glfwSwapBuffers(window);
	glfwPollEvents();
}

void Renderer::processInput()
{
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
	{
		RT->test();
	}

	if ((glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) &&
		(glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS))
	{
		RT->save();
	}
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}

