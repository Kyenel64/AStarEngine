#include <../Hoth.h>

int main()
{
	int width = 1920;
	int height = 1080;

	Renderer r(width, height, "Test Window", 4, 3);

	RayTracer* RT = new RayTracer(width, height);

	//RT->addObject(sphere);


	while (true)
	{
		r.Render(RT);
	}

	r.Terminate();

	return 0;
}