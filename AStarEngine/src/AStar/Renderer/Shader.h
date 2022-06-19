#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

class Shader
{
public:
	// Shader program ID
	unsigned int ID;

	// Default contructor reads and builds shader
	Shader(const char* vShaderCode, const char* fShaderCode);

	// Activate shader
	void use();
	// Utility uniform functions
	void setBool(const std::string& name, bool value) const;
	void setInt(const std::string& name, int value) const;
	void setFloat(const std::string& name, float value) const;
	void setMat4(const std::string& name, glm::mat4 value) const;
	void setVec3(const std::string& name, glm::vec3 value) const;

private:

	void checkCompileError(unsigned int shader, std::string type);
};