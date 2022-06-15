#include "Shader.h"

Shader::Shader(const char* vShaderCode, const char* fShaderCode)
{
	//// 1. Retrieve vertex and fragment source code from filepath
	//std::string vertexCode;
	//std::string fragmentCode;
	//std::ifstream vShaderFile;
	//std::ifstream fShaderFile;
	//// Make sure ifstream can throw exceptions
	//vShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	//fShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	//try
	//{
	//	// Open files
	//	vShaderFile.open(vertexPath);
	//	fShaderFile.open(fragmentPath);
	//	std::stringstream vShaderStream, fShaderStream;
	//	// Read files buffer contents into stream
	//	vShaderStream << vShaderFile.rdbuf();
	//	fShaderStream << fShaderFile.rdbuf();
	//	// Close file handlers
	//	vShaderFile.close();
	//	fShaderFile.close();
	//	// Convert stream into string
	//	vertexCode = vShaderStream.str();
	//	fragmentCode = fShaderStream.str();
	//}
	//catch (std::ifstream::failure e)
	//{
	//	std::cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ" << std::endl;
	//}
	//const char* vShaderCode = vertexCode.c_str();
	//const char* fShaderCode = fragmentCode.c_str();


	// 2. Compile shaders
	unsigned int vertex, fragment;

	// Vertex shader
	vertex = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertex, 1, &vShaderCode, NULL);
	glCompileShader(vertex);
	checkCompileError(vertex, "VERTEX");

	// Fragment shader
	fragment = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragment, 1, &fShaderCode, NULL);
	glCompileShader(fragment);
	checkCompileError(fragment, "FRAGMENT");

	// Shader Program
	ID = glCreateProgram();
	glAttachShader(ID, vertex);
	glAttachShader(ID, fragment);
	glLinkProgram(ID);
	checkCompileError(ID, "PROGRAM");


	// Delete shaders; they're linked into program so no longer needed
	glDeleteShader(vertex);
	glDeleteShader(fragment);
}

void Shader::use()
{
	glUseProgram(ID);
}

void Shader::setBool(const std::string& name, bool value) const
{
	glUniform1i(glGetUniformLocation(ID, name.c_str()), (int)value);
}

void Shader::setInt(const std::string& name, int value) const
{
	glUniform1i(glGetUniformLocation(ID, name.c_str()), value);
}

void Shader::setFloat(const std::string& name, float value) const
{
	glUniform1f(glGetUniformLocation(ID, name.c_str()), value);
}

void Shader::setMat4(const std::string& name, glm::mat4 value) const
{
	glUniformMatrix4fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, glm::value_ptr(value));
}

void Shader::setVec3(const std::string& name, glm::vec3 value) const
{
	glUniform3fv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]);
}

void Shader::checkCompileError(unsigned int shader, std::string type)
{
	int success;
	char infoLog[1024];
	if (type != "PROGRAM")
	{
		glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
		if (!success)
		{
			glGetShaderInfoLog(shader, 1024, NULL, infoLog);
			std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
		}
	}
	else
	{
		glGetProgramiv(shader, GL_LINK_STATUS, &success);
		if (!success)
		{
			glGetProgramInfoLog(shader, 1024, NULL, infoLog);
			std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
		}
	}
}