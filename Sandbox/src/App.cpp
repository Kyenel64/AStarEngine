#include <../Hoth.h>

int main()
{
	int width = 1920;
	int height = 1080;

	Renderer r(width, height, "Test Window", 4, 3);

	RayTracer* RT = new RayTracer;


	while (true)
	{
		r.Render(RT);
	}

	glfwTerminate();

	return 0;
}