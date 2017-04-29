/*
 * This software is Copyright (c) 2017 Sayantan Datta <std2048 at gmail dot com>
 * and it is hereby released to the general public under the following terms:
 * Redistribution and use in source and binary forms, with or without modification, are permitted for non-profit
 * and non-commericial purposes.
 */
#ifndef __OclCompute_h_
#define __OclCompute_h_
#include <CL/cl.hpp>
#include "DataType.h"

#define OCL_EXTRA_INFO 1
#define OCL_INCLUDE_PATH ""

class OclCompute {
	static void test();
	static std::string getErrorString(cl_int error);
	static std::string readSource(std::string fName);

	static std::vector<cl_platform_id> platforms;
	static std::vector<cl_device_id> devices; // List of devices from all available platforms

	/*
	 *http://stackoverflow.com/questions/30218434/opencl-one-program-running-one-multiple-devices
	 *
	 * OpenCL context can encapsulate multiple devices from one platform, each program can be compiled for
	 * all devices in a context or one can individually pass flags for each device in a program. To keep
	 * things simple, without loss of generality, we create context, programs per device.
	 */

	/* For each active device*/
	static std::vector<cl_device_id> activeDevices;
	static std::vector<cl_context> contexts; // Create context per device
	static std::vector<cl_command_queue> cmdQs; // Create command queue per device
	static std::vector<std::vector<cl_kernel>> kernels; // Multiple kernels per device
	static std::vector<cl_ulong> maxGlobalMemSz; // Store max global memory for each device
	static std::vector<cl_ulong> maxMemAllocSz; // Store max memory object size

	static void _0_checkDevices();
	static void _1_activateDevices(const std::vector<unsigned int> &devList);
	static void _2_initKernels();

	/* Application Specific*/
	static std::vector<cl_mem> clBufDeltaVel;
	static std::vector<cl_mem> clBufBodyIndex;
	static std::vector<cl_mem> clBufConstNormalD_A;
	static std::vector<cl_mem> clBufConstNormalM_A;
	static std::vector<cl_mem> clBufConstTangentD_A;
	static std::vector<cl_mem> clBufConstTangentM_A;
	static std::vector<cl_mem> clBufConstNormalD_B;
	static std::vector<cl_mem> clBufConstNormalM_B;
	static std::vector<cl_mem> clBufConstTangentD_B;
	static std::vector<cl_mem> clBufConstTangentM_B;
	static std::vector<cl_mem> clBufB;
	static std::vector<cl_mem> clBufLambda;
	static std::vector<cl_mem> clBufDeltaLambda;

	static unsigned int iterCount;
	static scalar mu;
	static void _3_createBuffer();
	static void _4_setKernelArgsStatic();
public:
	static void init(unsigned int iterCount, scalar mu);

	static void _0_run(unsigned int nBody, unsigned int nContacts,
				std::vector<vec6> &deltaVel, const std::vector<ivec2> &bodyIndex,
				const std::vector<vec6> &bufConstNormalD_A, const std::vector<vec6> &bufConstNormalM_A,
				const std::vector<vec6> &bufConstTangentD_A, const std::vector<vec6> &bufConstTangentM_A,
				const std::vector<vec6> &bufConstNormalD_B, const std::vector<vec6> &bufConstNormalM_B,
				const std::vector<vec6> &bufConstTangentD_B, const std::vector<vec6> &bufConstTangentM_B,
				const std::vector<vec2> &bufB, std::vector<vec2> &bufLambda);
};

#define HANDLE_CLERROR(cl_error, message)	  \
	do { cl_int __err = (cl_error); \
		if (__err != CL_SUCCESS) { \
			std::cout<<"OpenCL "<<OclCompute::getErrorString(__err)<<" error in "<<	\
				__FILE__<<":"<<__LINE__<<" - "<<message<<std::endl; \
			exit(0); \
		} \
	} while (0)

#endif
