/*
 * This software is Copyright (c) 2017 Sayantan Datta <std2048 at gmail dot com>
 * and it is hereby released to the general public under the following terms:
 * Redistribution and use in source and binary forms, with or without modification, are permitted for non-profit
 * and non-commericial purposes.
 */
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
