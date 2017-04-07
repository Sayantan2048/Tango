#ifndef __DataType_h_
#define __DataType_h_
#include <vec3.hpp>
//#define DP // double precision

#ifdef DP

typedef glm::dvec3 vec3;
typedef double scalar;

#else

typedef glm::vec3 vec3;
typedef float scalar;

#endif

struct vec6 {
	vec3 vLin;
	vec3 vAng;
};

struct ivec2 {
	unsigned int indexA;
	unsigned int indexB;
};

struct vec2 {
	scalar s1;
	scalar s2;
};

#endif
