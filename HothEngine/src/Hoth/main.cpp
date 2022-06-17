#include <Hoth/Renderer/Renderer.h>

int main()
{
	Data data;

	Renderer r(data, "Test Window", 4, 3);

	RayTracer* RT = new RayTracer(data);


	while (true)
	{
		r.Render(RT);
	}

	r.Terminate();

	return 0;
}