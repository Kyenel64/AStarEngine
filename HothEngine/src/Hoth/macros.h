#pragma once

#ifdef HT_PLATFORM_WINDOWS
	#ifdef HT_BUILD_DLL
		#define HOTH_API __declspec(dllexport)
	#else
		#define HOTH_API __declspec(dllimport)
	#endif
#else
	#error Hoth only supports windows
#endif
