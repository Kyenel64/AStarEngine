#include "Renderer.h"

#include <iostream>
#include <vector>

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
std::vector<float> toVector(vec3 vec);
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
void mouse_callback(GLFWwindow* window, double xposIn, double yposIn);

float lastFrame = 0.0;
float deltaTime = 0.0;

float lastX = 1920 / 2.0f;
float lastY = 1080 / 2.0f;
bool firstMouse = true;

Renderer::Renderer(RayTracer* RT, Data* data, std::string title, int GLVERSION_MAJOR, 
	int GLVERSION_MINOR) : RT(RT), data(data)
{
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, GLVERSION_MAJOR);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, GLVERSION_MINOR);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_MAXIMIZED, GLFW_FALSE); // Maximized window

	window = glfwCreateWindow(data->image_width, data->image_height, title.c_str(), NULL, NULL);

	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
	}

	// Create pointer to Renderer in GLFW context
	glfwSetWindowUserPointer(window, this);

	glfwMakeContextCurrent(window);
	glfwSetKeyCallback(window, key_callback);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwSetCursorPosCallback(window, mouse_callback);

	// Initialize GLAD
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
	}

	// Set viewport inside the window
	glViewport(0, 0, data->image_width, data->image_height);
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

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
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);


}

Renderer::~Renderer()
{
	
}

void Renderer::Render()
{
	processInput(window);

	// calculate deltaTime
	float currentFrame = glfwGetTime();
	deltaTime = currentFrame - lastFrame;
	lastFrame = currentFrame;

	if (glfwWindowShouldClose(window))
		glfwTerminate();

	// update Frame
	if (RT->GenerateFrame())
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, data->image_width, data->image_height, 0, GL_RGB, GL_UNSIGNED_BYTE, RT->getFrame());
	else
		std::cout << "Failed to load texture" << std::endl;


	// render frame
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);


	glfwSwapBuffers(window);
	glfwPollEvents();
}

RayTracer* Renderer::getRayTracer() const
{
	return RT;
}

// dont implement until resizing works properly
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	//glViewport(0, 0, width, height);
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	// Get pointer to Renderer and RayTracer classes
	Renderer* RD = static_cast<Renderer*>(glfwGetWindowUserPointer(window));
	RayTracer* RT = RD->getRayTracer();

	if (key == GLFW_KEY_S && mods == GLFW_MOD_CONTROL && action == GLFW_PRESS)
	{
		RT->save();
		RD->serialize();
	}

	if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
	{
		//
	}

	if (key == GLFW_KEY_X && action == GLFW_PRESS)
	{
		RT->setRenderMode();
	}
}

void Renderer::processInput(GLFWwindow* window)
{
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		RT->move(FORWARD, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		RT->move(BACKWARD, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		RT->move(LEFT, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		RT->move(RIGHT, deltaTime);
}

void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
{
	Renderer* RD = static_cast<Renderer*>(glfwGetWindowUserPointer(window));
	RayTracer* RT = RD->getRayTracer();
	float xpos = (float)(xposIn);
	float ypos = (float)(yposIn);

	// Prevent frame skip on first mouse input
	if (firstMouse)
	{
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}

	float xoffset = xpos - lastX;
	float yoffset = lastY - ypos;

	lastX = xpos;
	lastY = ypos;

	RT->mouseMove(xoffset, yoffset);

}

// Output data to json file.
void Renderer::serialize()
{
	json j;
	data = RT->getData();

	// Image properties
	j["title"] = data->title;
	j["aspect_ratio"] = data->aspect_ratio;
	j["image_width"] = data->image_width;
	j["image_height"] = data->image_height;

	// Rendering properties
	j["samples_per_pixel"] = data->samples_per_pixel;
	j["max_depth"] = data->max_depth;
	
	// Camera properties
	j["fov"] = data->fov;
	j["focal_length"] = data->focal_length;
	j["origin"] = { data->origin.x(), data->origin.y(), data->origin.z() };
	j["lookAt"] = { data->lookAt.x(), data->lookAt.y(), data->lookAt.z() };
	j["up"] = { data->up.x(), data->up.y(), data->up.z() };
	j["dist_to_focus"] = data->dist_to_focus;
	j["aperture"] = data->aperture;

	// Object properties
	j["objectCount"] = data->objectCount;
	for (int i = 0; i < data->objectCount; i++)
	{
		j["objectData"][i][0] = data->objData[i].id;
		j["objectData"][i][1] = { data->objData[i].Pos.x(), data->objData[i].Pos.y(), data->objData[i].Pos.z() };
		j["objectData"][i][2] = data->objData[i].radius;
		j["objectData"][i][3] = data->objData[i].matID;
	}

	// Material properties
	j["materialCount"] = data->materialCount;
	for (int i = 0; i < data->materialCount; i++)
	{
		j["materialData"][i][0] = data->matData[i].id;
		j["materialData"][i][1] = { data->matData[i].Col.x(), data->matData[i].Col.y(), data->matData[i].Col.z() };
		j["materialData"][i][2] = data->matData[i].matType;
	}

	std::ofstream o("../save/save.astar");
	o << std::setw(4) << j << std::endl;
}

std::vector<float> toVector(vec3 vec)
{
	std::vector<float> v;
	v.push_back(vec.x());
	v.push_back(vec.y());
	v.push_back(vec.z());
	return v;
}