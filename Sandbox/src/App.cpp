#include <../Hoth.h>

int main()
{
	int width = 1920;
	int height = 1080;

	Renderer r(width, height, "Test Window", 4, 3);
	r.Start();

	return 0;
}