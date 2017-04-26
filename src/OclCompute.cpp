#include "OclCompute.h"
#include <iostream>
#include <fstream>
#include <streambuf>
#include <string>

// Bunch of static variables
std::vector<cl_platform_id> OclCompute::platforms;
std::vector<cl_device_id> OclCompute::devices;
std::vector<cl_device_id> OclCompute::activeDevices;
std::vector<cl_context> OclCompute::contexts;
std::vector<cl_command_queue> OclCompute::cmdQs;
std::vector<std::vector<cl_kernel>> OclCompute::kernels;
std::vector<unsigned long> OclCompute::maxGlobalMemSz;
std::vector<unsigned long> OclCompute::maxMemAllocSz;
unsigned int OclCompute::iterCount;
scalar OclCompute::mu;


std::vector<cl_mem> OclCompute::clBufDeltaVel;
std::vector<cl_mem> OclCompute::clBufBodyIndex;
std::vector<cl_mem> OclCompute::clBufConstNormalD_A;
std::vector<cl_mem> OclCompute::clBufConstNormalM_A;
std::vector<cl_mem> OclCompute::clBufConstTangentD_A;
std::vector<cl_mem> OclCompute::clBufConstTangentM_A;
std::vector<cl_mem> OclCompute::clBufConstNormalD_B;
std::vector<cl_mem> OclCompute::clBufConstNormalM_B;
std::vector<cl_mem> OclCompute::clBufConstTangentD_B;
std::vector<cl_mem> OclCompute::clBufConstTangentM_B;
std::vector<cl_mem> OclCompute::clBufB;
std::vector<cl_mem> OclCompute::clBufLambda;
std::vector<cl_mem> OclCompute::clBufDeltaLambda;

void OclCompute::test() {
	cl_platform_id platform;
	cl_device_id dev;
	int err;

	/* Identify a platform */
	err = clGetPlatformIDs(1, &platform, NULL);
	if(err < 0) {
		perror("Couldn't identify a platform");
	    exit(1);
	}

	/* Access a device */
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
	if(err == CL_DEVICE_NOT_FOUND) {
		err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
	}
	if(err < 0) {
		perror("Couldn't access any devices");
	    exit(1);
	}

	cl_context_properties properties[3];
	properties[0] = CL_CONTEXT_PLATFORM;
	properties[1] = (cl_context_properties)platform;
	properties[2] = 0;

	cl_context context;
	context = clCreateContext(properties, 1, &dev, NULL, NULL, &err);
	HANDLE_CLERROR(err, "Error Building context");

	cl_command_queue cmdq;
	cmdq = clCreateCommandQueue(context, dev, 0, &err);
	HANDLE_CLERROR(err, "Error Building CommandQueue");

	cl_mem buf;
	buf = clCreateBuffer(context, CL_MEM_READ_WRITE, 32 * 1024 * 1024, NULL, &err);
	HANDLE_CLERROR(err, "Error Building Buffer");

	vec6 A;
	HANDLE_CLERROR(clEnqueueWriteBuffer(cmdq, buf, CL_TRUE, 0, sizeof(vec6), &A, 0, NULL, NULL), "Error writing to buffer");

	std::cout<<"Test Passed"<<std::endl;
}

void OclCompute::_0_checkDevices() {
	test();
	char infoStr[1024];
	cl_uint numPlatforms;
	HANDLE_CLERROR(clGetPlatformIDs(0, NULL, &numPlatforms),
			"Error querying Platforms.");
	platforms.resize(numPlatforms);
	HANDLE_CLERROR(clGetPlatformIDs(numPlatforms, &platforms[0], NULL),
			"Error querying Platforms.");

	int ctr = 0;
	std::cout<<"Devices found:"<<std::endl;
	for (size_t i = 0; i < platforms.size(); i++) {
		cl_uint numDevices;
		HANDLE_CLERROR(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL,
				0, NULL, &numDevices), "Error querying Devices.");
		devices.resize(devices.size() + numDevices);
		HANDLE_CLERROR(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL,
				numDevices, &devices[devices.size() - numDevices], NULL), "Error querying Devices.");

		for (size_t j = devices.size() - numDevices; j < devices.size(); j++) {
			std::cout<<"Device No: "<<ctr++;
			HANDLE_CLERROR(clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME,
					sizeof(infoStr), infoStr, NULL), "Error querying CL_PLATFORM_NAME");
			std::cout<<", Platform: "<<infoStr;
			HANDLE_CLERROR(clGetDeviceInfo(devices[j], CL_DEVICE_NAME,
					sizeof(infoStr), infoStr, NULL), "Error querying CL_DEVICE_NAME");
			std::cout<<", Device: "<<infoStr;
#if OCL_EXTRA_INFO
			cl_ulong bytes;
			HANDLE_CLERROR(clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE,
					sizeof(bytes), &bytes, NULL), "Error querying CL_DEVICE_GLOBAL_MEM_SIZE");
			std::cout<<", Global Mem Size: "<<bytes;
			HANDLE_CLERROR(clGetDeviceInfo(devices[j], CL_DEVICE_MAX_MEM_ALLOC_SIZE,
					sizeof(bytes), &bytes, NULL), "Error querying CL_DEVICE_MAX_MEM_ALLOC_SIZE");
			std::cout<<", Mem Alloc Size: "<<bytes;
#endif
			std::cout<<std::endl;
		}
	}
}


void OclCompute::_1_activateDevices(const std::vector<unsigned int> &devList) {
	cl_int err;
	char infoStr[1024];

	for (size_t i = 0; i < devList.size(); i++)
		activeDevices.push_back(devices[devList[i]]);

	std::cout<<std::endl<<"Device(s) in use:"<<std::endl;

	for (size_t i = 0; i < activeDevices.size(); i++) {
		cl_context_properties properties[3];
		cl_platform_id tempPlatformID;
		HANDLE_CLERROR(clGetDeviceInfo(activeDevices[i], CL_DEVICE_PLATFORM,
				sizeof(tempPlatformID), &tempPlatformID, NULL), "Error querying CL_DEVICE_PLATFORM");
		properties[0] = CL_CONTEXT_PLATFORM;
		properties[1] = (cl_context_properties)tempPlatformID;
		properties[2] = 0;

		contexts.push_back(clCreateContext(properties, 1, &activeDevices[i], NULL, NULL, &err));
		HANDLE_CLERROR(err, "Failed to create Context.");

		cmdQs.push_back(clCreateCommandQueue(contexts[contexts.size() - 1], activeDevices[i], 0, &err));
		HANDLE_CLERROR(err, "Failed to create Command Queue.");

		cl_ulong bytes;
		HANDLE_CLERROR(clGetDeviceInfo(activeDevices[i], CL_DEVICE_GLOBAL_MEM_SIZE,
				sizeof(bytes), &bytes, NULL), "Error querying CL_DEVICE_GLOBAL_MEM_SIZE");
		maxGlobalMemSz.push_back(bytes);
		HANDLE_CLERROR(clGetDeviceInfo(activeDevices[i], CL_DEVICE_MAX_MEM_ALLOC_SIZE,
				sizeof(bytes), &bytes, NULL), "Error querying CL_DEVICE_MAX_MEM_ALLOC_SIZE");
		maxMemAllocSz.push_back(bytes);

		HANDLE_CLERROR(clGetDeviceInfo(activeDevices[i], CL_DEVICE_NAME,
				sizeof(infoStr), infoStr, NULL), "Error querying CL_DEVICE_NAME");
		std::cout<<"Device: "<<infoStr<<std::endl;
	}
}

void OclCompute::_2_initKernels() {
	cl_int err;

	for (size_t i = 0; i < activeDevices.size(); i++) {
		std::vector<cl_kernel> kernelList;
		do {
			std::string kernelSource = readSource("kernel/jacobi.cl");
			if (kernelSource.size() == 0) {
				// Release resources
				exit(0);
			}
			cl::Program::Sources sources;
			sources.push_back({kernelSource.c_str(), kernelSource.length()});
			do {
				const char *srcPtr[] = {kernelSource.c_str()};
				cl_program program = clCreateProgramWithSource(contexts[i], 1, srcPtr, NULL, &err);
				HANDLE_CLERROR(err, "Failed to create Program.");

				std::string build_opts;
				if (std::string(OCL_INCLUDE_PATH) != "")
					build_opts = std::string("-I ") + std::string(OCL_INCLUDE_PATH);
// Add build opts

				build_opts += "-D ITER_COUNT=" + std::to_string(iterCount) +
						" -D MU=" + std::to_string(mu);

				cl_int build_code = clBuildProgram(program, 0, NULL,
						build_opts.c_str(), NULL, NULL);

				size_t logSize;
				HANDLE_CLERROR(clGetProgramBuildInfo(program,
					                                 activeDevices[i],
					                                 CL_PROGRAM_BUILD_LOG, 0, NULL,
					                                 &logSize), "Error while getting build info.");
				char *build_log = new char[logSize + 1];

				HANDLE_CLERROR(clGetProgramBuildInfo(program,
													 activeDevices[i],
					                                 CL_PROGRAM_BUILD_LOG, logSize + 1,
					                                 (void *)build_log, NULL), "Error while getting build info");

				// Report build errors and warnings
				if ((build_code != CL_SUCCESS)) {
					// Give us much info about error and exit
					std::cerr<<"Options used: "<<build_opts<<"\n"<<kernelSource
							<<"\n"<<build_log<<std::endl;
					HANDLE_CLERROR(build_code, "clBuildProgram failed.");
				}

				kernelList.push_back(clCreateKernel(program, "clearBuffer", &err));
				HANDLE_CLERROR(err, "Failed to build kernel.");

				kernelList.push_back(clCreateKernel(program, "jacobi_parallel", &err));
				HANDLE_CLERROR(err, "Failed to build kernel.");

				kernelList.push_back(clCreateKernel(program, "jacobi_serial", &err));
				HANDLE_CLERROR(err, "Failed to build kernel.");

				kernelList.push_back(clCreateKernel(program, "jacobi_comb", &err));
				HANDLE_CLERROR(err, "Failed to build kernel.");

				kernelList.push_back(clCreateKernel(program, "jacobi_v2", &err));
				HANDLE_CLERROR(err, "Failed to build kernel.");

				kernelList.push_back(clCreateKernel(program, "jacobi_v3", &err));
				HANDLE_CLERROR(err, "Failed to build kernel.");

				kernelList.push_back(clCreateKernel(program, "jacobi_v3_split1", &err));
				HANDLE_CLERROR(err, "Failed to build kernel.");

				kernelList.push_back(clCreateKernel(program, "jacobi_v3_split2", &err));
				HANDLE_CLERROR(err, "Failed to build kernel.");

				kernelList.push_back(clCreateKernel(program, "jacobi_comb_v4", &err));
				HANDLE_CLERROR(err, "Failed to build kernel.");

				kernelList.push_back(clCreateKernel(program, "jacobi_norm", &err));
				HANDLE_CLERROR(err, "Failed to build kernel.");

				HANDLE_CLERROR(clReleaseProgram(program), "Failed to release Program.");
			} while(0);

		} while(0);
		kernels.push_back(kernelList);
	}
}

/* Dumb way to create buffers!!*/
void OclCompute::_3_createBuffer() {
	cl_int err;

	for (size_t i = 0; i < activeDevices.size(); i++) {
		clBufDeltaVel.push_back(clCreateBuffer(contexts[i], CL_MEM_READ_WRITE, 32 * 1024 * 1024, NULL, &err));
		HANDLE_CLERROR(err, "Failed to create Buffer.");

		clBufBodyIndex.push_back(clCreateBuffer(contexts[i], CL_MEM_READ_ONLY, 8 * 1024 * 1024, NULL, &err));
		HANDLE_CLERROR(err, "Failed to create Buffer.");
		clBufConstNormalD_A.push_back(clCreateBuffer(contexts[i], CL_MEM_READ_ONLY, 32 * 1024 * 1024, NULL, &err));
		HANDLE_CLERROR(err, "Failed to create Buffer.");
		clBufConstNormalM_A.push_back(clCreateBuffer(contexts[i], CL_MEM_READ_ONLY, 32 * 1024 * 1024, NULL, &err));
		HANDLE_CLERROR(err, "Failed to create Buffer.");
		clBufConstTangentD_A.push_back(clCreateBuffer(contexts[i], CL_MEM_READ_ONLY, 32 * 1024 * 1024, NULL, &err));
		HANDLE_CLERROR(err, "Failed to create Buffer.");
		clBufConstTangentM_A.push_back(clCreateBuffer(contexts[i], CL_MEM_READ_ONLY, 32 * 1024 * 1024, NULL, &err));
		HANDLE_CLERROR(err, "Failed to create Buffer.");

		clBufConstNormalD_B.push_back(clCreateBuffer(contexts[i], CL_MEM_READ_ONLY, 32 * 1024 * 1024, NULL, &err));
		HANDLE_CLERROR(err, "Failed to create Buffer.");
		clBufConstNormalM_B.push_back(clCreateBuffer(contexts[i], CL_MEM_READ_ONLY, 32 * 1024 * 1024, NULL, &err));
		HANDLE_CLERROR(err, "Failed to create Buffer.");
		clBufConstTangentD_B.push_back(clCreateBuffer(contexts[i], CL_MEM_READ_ONLY, 32 * 1024 * 1024, NULL, &err));
		HANDLE_CLERROR(err, "Failed to create Buffer.");
		clBufConstTangentM_B.push_back(clCreateBuffer(contexts[i], CL_MEM_READ_ONLY, 32 * 1024 * 1024, NULL, &err));
		HANDLE_CLERROR(err, "Failed to create Buffer.");

		clBufB.push_back(clCreateBuffer(contexts[i], CL_MEM_READ_ONLY, 32 * 1024 * 1024, NULL, &err));
		HANDLE_CLERROR(err, "Failed to create Buffer.");

		clBufLambda.push_back(clCreateBuffer(contexts[i], CL_MEM_READ_WRITE, 32 * 1024 * 1024, NULL, &err));
		HANDLE_CLERROR(err, "Failed to create Buffer.");
		clBufDeltaLambda.push_back(clCreateBuffer(contexts[i], CL_MEM_READ_WRITE, 32 * 1024 * 1024, NULL, &err));
		HANDLE_CLERROR(err, "Failed to create Buffer.");
	}
}

void OclCompute::_4_setKernelArgsStatic() {
	for (size_t i = 0; i < activeDevices.size(); i++) {
		int ctr = 0;
		HANDLE_CLERROR(clSetKernelArg(kernels[i][0], ctr++, sizeof(cl_mem), &clBufDeltaVel[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][0], ctr++, sizeof(cl_mem), &clBufLambda[i]), "Failed to set kernel args.");

		ctr = 0;
		HANDLE_CLERROR(clSetKernelArg(kernels[i][1], ctr++, sizeof(cl_mem), &clBufDeltaVel[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][1], ctr++, sizeof(cl_mem), &clBufBodyIndex[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][1], ctr++, sizeof(cl_mem), &clBufConstNormalD_A[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][1], ctr++, sizeof(cl_mem), &clBufConstTangentD_A[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][1], ctr++, sizeof(cl_mem), &clBufConstNormalD_B[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][1], ctr++, sizeof(cl_mem), &clBufConstTangentD_B[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][1], ctr++, sizeof(cl_mem), &clBufB[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][1], ctr++, sizeof(cl_mem), &clBufLambda[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][1], ctr++, sizeof(cl_mem), &clBufDeltaLambda[i]), "Failed to set kernel args.");

		ctr = 0;
		HANDLE_CLERROR(clSetKernelArg(kernels[i][2], ctr++, sizeof(cl_mem), &clBufDeltaVel[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][2], ctr++, sizeof(cl_mem), &clBufBodyIndex[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][2], ctr++, sizeof(cl_mem), &clBufConstNormalM_A[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][2], ctr++, sizeof(cl_mem), &clBufConstTangentM_A[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][2], ctr++, sizeof(cl_mem), &clBufConstNormalM_B[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][2], ctr++, sizeof(cl_mem), &clBufConstTangentM_B[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][2], ctr++, sizeof(cl_mem), &clBufDeltaLambda[i]), "Failed to set kernel args.");

		ctr = 0;
		HANDLE_CLERROR(clSetKernelArg(kernels[i][3], ctr++, sizeof(cl_mem), &clBufDeltaVel[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][3], ctr++, sizeof(cl_mem), &clBufBodyIndex[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][3], ctr++, sizeof(cl_mem), &clBufConstNormalD_A[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][3], ctr++, sizeof(cl_mem), &clBufConstTangentD_A[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][3], ctr++, sizeof(cl_mem), &clBufConstNormalD_B[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][3], ctr++, sizeof(cl_mem), &clBufConstTangentD_B[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][3], ctr++, sizeof(cl_mem), &clBufConstNormalM_A[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][3], ctr++, sizeof(cl_mem), &clBufConstTangentM_A[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][3], ctr++, sizeof(cl_mem), &clBufConstNormalM_B[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][3], ctr++, sizeof(cl_mem), &clBufConstTangentM_B[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][3], ctr++, sizeof(cl_mem), &clBufB[i]), "Failed to set kernel args.");

		ctr = 0;
		HANDLE_CLERROR(clSetKernelArg(kernels[i][4], ctr++, sizeof(cl_mem), &clBufDeltaVel[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][4], ctr++, sizeof(cl_mem), &clBufBodyIndex[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][4], ctr++, sizeof(cl_mem), &clBufConstNormalD_A[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][4], ctr++, sizeof(cl_mem), &clBufConstTangentD_A[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][4], ctr++, sizeof(cl_mem), &clBufConstNormalD_B[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][4], ctr++, sizeof(cl_mem), &clBufConstTangentD_B[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][4], ctr++, sizeof(cl_mem), &clBufConstNormalM_A[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][4], ctr++, sizeof(cl_mem), &clBufConstTangentM_A[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][4], ctr++, sizeof(cl_mem), &clBufConstNormalM_B[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][4], ctr++, sizeof(cl_mem), &clBufConstTangentM_B[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][4], ctr++, sizeof(cl_mem), &clBufB[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][4], ctr++, sizeof(cl_mem), &clBufLambda[i]), "Failed to set kernel args.");

		ctr = 0;
		HANDLE_CLERROR(clSetKernelArg(kernels[i][5], ctr++, sizeof(cl_mem), &clBufDeltaVel[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][5], ctr++, sizeof(cl_mem), &clBufBodyIndex[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][5], ctr++, sizeof(cl_mem), &clBufConstNormalD_A[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][5], ctr++, sizeof(cl_mem), &clBufConstTangentD_A[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][5], ctr++, sizeof(cl_mem), &clBufConstNormalD_B[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][5], ctr++, sizeof(cl_mem), &clBufConstTangentD_B[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][5], ctr++, sizeof(cl_mem), &clBufConstNormalM_A[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][5], ctr++, sizeof(cl_mem), &clBufConstTangentM_A[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][5], ctr++, sizeof(cl_mem), &clBufConstNormalM_B[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][5], ctr++, sizeof(cl_mem), &clBufConstTangentM_B[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][5], ctr++, sizeof(cl_mem), &clBufB[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][5], ctr++, sizeof(cl_mem), &clBufLambda[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][5], ctr++, sizeof(cl_mem), &clBufDeltaLambda[i]), "Failed to set kernel args.");

		ctr = 0;
		HANDLE_CLERROR(clSetKernelArg(kernels[i][6], ctr++, sizeof(cl_mem), &clBufBodyIndex[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][6], ctr++, sizeof(cl_mem), &clBufConstNormalD_A[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][6], ctr++, sizeof(cl_mem), &clBufConstTangentD_A[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][6], ctr++, sizeof(cl_mem), &clBufConstNormalD_B[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][6], ctr++, sizeof(cl_mem), &clBufConstTangentD_B[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][6], ctr++, sizeof(cl_mem), &clBufConstNormalM_A[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][6], ctr++, sizeof(cl_mem), &clBufConstTangentM_A[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][6], ctr++, sizeof(cl_mem), &clBufConstNormalM_B[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][6], ctr++, sizeof(cl_mem), &clBufConstTangentM_B[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][6], ctr++, sizeof(cl_mem), &clBufDeltaLambda[i]), "Failed to set kernel args.");

		ctr = 0;
		HANDLE_CLERROR(clSetKernelArg(kernels[i][7], ctr++, sizeof(cl_mem), &clBufDeltaVel[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][7], ctr++, sizeof(cl_mem), &clBufBodyIndex[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][7], ctr++, sizeof(cl_mem), &clBufConstNormalM_A[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][7], ctr++, sizeof(cl_mem), &clBufConstTangentM_A[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][7], ctr++, sizeof(cl_mem), &clBufConstNormalM_B[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][7], ctr++, sizeof(cl_mem), &clBufConstTangentM_B[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][7], ctr++, sizeof(cl_mem), &clBufB[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][7], ctr++, sizeof(cl_mem), &clBufLambda[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][7], ctr++, sizeof(cl_mem), &clBufDeltaLambda[i]), "Failed to set kernel args.");

		ctr = 0;
		HANDLE_CLERROR(clSetKernelArg(kernels[i][8], ctr++, sizeof(cl_mem), &clBufDeltaVel[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][8], ctr++, sizeof(cl_mem), &clBufBodyIndex[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][8], ctr++, sizeof(cl_mem), &clBufConstNormalD_A[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][8], ctr++, sizeof(cl_mem), &clBufConstTangentD_A[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][8], ctr++, sizeof(cl_mem), &clBufConstNormalD_B[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][8], ctr++, sizeof(cl_mem), &clBufConstTangentD_B[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][8], ctr++, sizeof(cl_mem), &clBufConstNormalM_A[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][8], ctr++, sizeof(cl_mem), &clBufConstTangentM_A[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][8], ctr++, sizeof(cl_mem), &clBufConstNormalM_B[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][8], ctr++, sizeof(cl_mem), &clBufConstTangentM_B[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][8], ctr++, sizeof(cl_mem), &clBufB[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][8], ctr++, sizeof(cl_mem), &clBufLambda[i]), "Failed to set kernel args.");

		ctr = 0;
		HANDLE_CLERROR(clSetKernelArg(kernels[i][9], ctr++, sizeof(cl_mem), &clBufBodyIndex[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][9], ctr++, sizeof(cl_mem), &clBufConstNormalD_A[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][9], ctr++, sizeof(cl_mem), &clBufConstTangentD_A[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][9], ctr++, sizeof(cl_mem), &clBufConstNormalD_B[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][9], ctr++, sizeof(cl_mem), &clBufConstTangentD_B[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][9], ctr++, sizeof(cl_mem), &clBufConstNormalM_A[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][9], ctr++, sizeof(cl_mem), &clBufConstTangentM_A[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][9], ctr++, sizeof(cl_mem), &clBufConstNormalM_B[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][9], ctr++, sizeof(cl_mem), &clBufConstTangentM_B[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][9], ctr++, sizeof(cl_mem), &clBufDeltaLambda[i]), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][9], ctr++, sizeof(cl_mem), &clBufB[i]), "Failed to set kernel args.");
	}
}

void show6(vec6 v) {
	std::cout<<v.vLin.x<<" "<<v.vLin.y<<" "<<v.vLin.z<<" "<<v.vAng.x<<" "<<v.vAng.y<<" "<<v.vAng.z<<std::endl;
}

void OclCompute::_0_run(unsigned int nBody, unsigned int nContacts,
			std::vector<vec6> &deltaVel, const std::vector<ivec2> &bodyIndex,
			const std::vector<vec6> &bufConstNormalD_A, const std::vector<vec6> &bufConstNormalM_A,
			const std::vector<vec6> &bufConstTangentD_A, const std::vector<vec6> &bufConstTangentM_A,
			const std::vector<vec6> &bufConstNormalD_B, const std::vector<vec6> &bufConstNormalM_B,
			const std::vector<vec6> &bufConstTangentD_B, const std::vector<vec6> &bufConstTangentM_B,
			const std::vector<vec2> &bufB, std::vector<vec2> &bufLambda) {

	for (size_t i = 0; i < activeDevices.size(); i++) {
		HANDLE_CLERROR(clEnqueueWriteBuffer(cmdQs[i], clBufBodyIndex[i], CL_FALSE, 0, sizeof(ivec2) * nContacts , &bodyIndex[0], 0, NULL, NULL), "Error writing to buffer.");

		HANDLE_CLERROR(clEnqueueWriteBuffer(cmdQs[i], clBufConstNormalD_A[i], CL_FALSE, 0, sizeof(vec6) * nContacts , &bufConstNormalD_A[0], 0, NULL, NULL), "Error writing to buffer.");
		HANDLE_CLERROR(clEnqueueWriteBuffer(cmdQs[i], clBufConstNormalM_A[i], CL_FALSE, 0, sizeof(vec6) * nContacts , &bufConstNormalM_A[0], 0, NULL, NULL), "Error writing to buffer.");
		HANDLE_CLERROR(clEnqueueWriteBuffer(cmdQs[i], clBufConstTangentD_A[i], CL_FALSE, 0, sizeof(vec6) * nContacts , &bufConstTangentD_A[0], 0, NULL, NULL), "Error writing to buffer.");
		HANDLE_CLERROR(clEnqueueWriteBuffer(cmdQs[i], clBufConstTangentM_A[i], CL_FALSE, 0, sizeof(vec6) * nContacts , &bufConstTangentM_A[0], 0, NULL, NULL), "Error writing to buffer.");

		HANDLE_CLERROR(clEnqueueWriteBuffer(cmdQs[i], clBufConstNormalD_B[i], CL_FALSE, 0, sizeof(vec6) * nContacts , &bufConstNormalD_B[0], 0, NULL, NULL), "Error writing to buffer.");
		HANDLE_CLERROR(clEnqueueWriteBuffer(cmdQs[i], clBufConstNormalM_B[i], CL_FALSE, 0, sizeof(vec6) * nContacts , &bufConstNormalM_B[0], 0, NULL, NULL), "Error writing to buffer.");
		HANDLE_CLERROR(clEnqueueWriteBuffer(cmdQs[i], clBufConstTangentD_B[i], CL_FALSE, 0, sizeof(vec6) * nContacts , &bufConstTangentD_B[0], 0, NULL, NULL), "Error writing to buffer.");
		HANDLE_CLERROR(clEnqueueWriteBuffer(cmdQs[i], clBufConstTangentM_B[i], CL_FALSE, 0, sizeof(vec6) * nContacts , &bufConstTangentM_B[0], 0, NULL, NULL), "Error writing to buffer.");

		HANDLE_CLERROR(clEnqueueWriteBuffer(cmdQs[i], clBufB[i], CL_TRUE, 0, sizeof(vec2) * nContacts , &bufB[0], 0, NULL, NULL), "Error writing to buffer.");

		scalar f = 0;
		HANDLE_CLERROR(clEnqueueFillBuffer(cmdQs[i], clBufDeltaVel[i], &f, sizeof(f), 0, sizeof(vec6) * nBody, 0, NULL, NULL), "Error filling buffer.");
		HANDLE_CLERROR(clEnqueueFillBuffer(cmdQs[i], clBufLambda[i], &f, sizeof(f), 0, sizeof(vec2) * nContacts, 0, NULL, NULL), "Error filling buffer.");

		//HANDLE_CLERROR(clSetKernelArg(kernels[i][4], 12, sizeof(cl_uint), &nContacts), "Failed to set kernel args.");
		//HANDLE_CLERROR(clSetKernelArg(kernels[i][4], 13, 2 * sizeof(uint) * nContacts, NULL), "Failed to set kernel args.");

		//HANDLE_CLERROR(clSetKernelArg(kernels[i][4], 14, 6 * sizeof(scalar) * nContacts, NULL), "Failed to set kernel args.");
		//HANDLE_CLERROR(clSetKernelArg(kernels[i][4], 15, 6 * sizeof(scalar) * nContacts, NULL), "Failed to set kernel args.");
		//HANDLE_CLERROR(clSetKernelArg(kernels[i][4], 16, 6 * sizeof(scalar) * nContacts, NULL), "Failed to set kernel args.");
		//HANDLE_CLERROR(clSetKernelArg(kernels[i][4], 17, 6 * sizeof(scalar) * nContacts, NULL), "Failed to set kernel args.");
		/*HANDLE_CLERROR(clSetKernelArg(kernels[i][0], 2, sizeof(cl_uint), &nBody), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][0], 3, sizeof(cl_uint), &nContacts), "Failed to set kernel args.");*/

		/*if (6 * nBody > 2 * nContacts)
			gws = 6 * nBody;
		else
			gws = 2 * nContacts;

		HANDLE_CLERROR(clEnqueueNDRangeKernel (cmdQs[i], kernels[i][0], 1, NULL, &gws, &lws, 0, NULL, NULL), "Failed to execute kernel");
*/
		size_t gws;
		size_t lws = 32;
		gws = nContacts;
		/*for (int j = 0; j < 500; j++) {
			HANDLE_CLERROR(clEnqueueNDRangeKernel (cmdQs[i], kernels[i][1], 1, NULL, &gws, &lws, 0, NULL, NULL), "Failed to execute kernel");
			HANDLE_CLERROR(clEnqueueNDRangeKernel (cmdQs[i], kernels[i][2], 1, NULL, &gws, &lws, 0, NULL, NULL), "Failed to execute kernel");
		}*/

		HANDLE_CLERROR(clSetKernelArg(kernels[i][3], 11, sizeof(cl_uint), &nContacts), "Failed to set kernel args.");
		HANDLE_CLERROR(clEnqueueNDRangeKernel (cmdQs[i], kernels[i][3], 1, NULL, &gws, &lws, 0, NULL, NULL), "Failed to execute kernel");
		//HANDLE_CLERROR(clEnqueueNDRangeKernel (cmdQs[i], kernels[i][4], 1, NULL, &gws, &lws, 0, NULL, NULL), "Failed to execute kernel");

		/*
		HANDLE_CLERROR(clSetKernelArg(kernels[i][5], 13, sizeof(cl_uint), &nContacts), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][5], 14, 2 * sizeof(uint) * nContacts, NULL), "Failed to set kernel args.");
        HANDLE_CLERROR(clSetKernelArg(kernels[i][5], 15, 6 * sizeof(scalar) * nContacts, NULL), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][5], 16, 6 * sizeof(scalar) * nContacts, NULL), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][5], 17, 6 * sizeof(scalar) * nContacts, NULL), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][5], 18, 6 * sizeof(scalar) * nContacts, NULL), "Failed to set kernel args.");
		HANDLE_CLERROR(clEnqueueNDRangeKernel (cmdQs[i], kernels[i][5], 1, NULL, &gws, &lws, 0, NULL, NULL), "Failed to execute kernel");
		*/
		/*HANDLE_CLERROR(clSetKernelArg(kernels[i][6], 10, sizeof(cl_uint), &nContacts), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][6], 11, 2 * sizeof(uint) * nContacts, NULL), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][6], 12, 6 * sizeof(scalar) * nContacts, NULL), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][6], 13, 6 * sizeof(scalar) * nContacts, NULL), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][6], 14, 6 * sizeof(scalar) * nContacts, NULL), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][6], 15, 6 * sizeof(scalar) * nContacts, NULL), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][7], 9, sizeof(cl_uint), &nContacts), "Failed to set kernel args.");
		HANDLE_CLERROR(clEnqueueNDRangeKernel (cmdQs[i], kernels[i][6], 1, NULL, &gws, &lws, 0, NULL, NULL), "Failed to execute kernel");
		HANDLE_CLERROR(clEnqueueNDRangeKernel (cmdQs[i], kernels[i][7], 1, NULL, &gws, &lws, 0, NULL, NULL), "Failed to execute kernel");
		*/
		//HANDLE_CLERROR(clEnqueueNDRangeKernel (cmdQs[i], kernels[i][8], 1, NULL, &gws, &lws, 0, NULL, NULL), "Failed to execute kernel");
		//HANDLE_CLERROR(clEnqueueReadBuffer(cmdQs[i], clBufLambda[i], CL_FALSE, 0, sizeof(vec2) * nContacts , &bufLambda[0], 0, NULL, NULL), "Error reading from buffer.");

		/*HANDLE_CLERROR(clSetKernelArg(kernels[i][9], 11, sizeof(cl_uint), &nContacts), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][9], 12, 2 * sizeof(uint) * nContacts, NULL), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][9], 13, 6 * sizeof(scalar) * nContacts, NULL), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][9], 14, 6 * sizeof(scalar) * nContacts, NULL), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][9], 15, 6 * sizeof(scalar) * nContacts, NULL), "Failed to set kernel args.");
		HANDLE_CLERROR(clSetKernelArg(kernels[i][9], 16, 6 * sizeof(scalar) * nContacts, NULL), "Failed to set kernel args.");

		HANDLE_CLERROR(clEnqueueNDRangeKernel (cmdQs[i], kernels[i][9], 1, NULL, &gws, &lws, 0, NULL, NULL), "Failed to execute kernel");
		HANDLE_CLERROR(clEnqueueNDRangeKernel (cmdQs[i], kernels[i][3], 1, NULL, &gws, &lws, 0, NULL, NULL), "Failed to execute kernel");
*/


		HANDLE_CLERROR(clEnqueueReadBuffer(cmdQs[i], clBufDeltaVel[i], CL_TRUE, 0, sizeof(vec6) * nBody , &deltaVel[0], 0, NULL, NULL), "Error reading from buffer.");
	}
}



void OclCompute::init(unsigned int iter, scalar frictionCoeff) {
	iterCount = iter;
	mu = frictionCoeff;

	_0_checkDevices();

	std::vector<unsigned int> devList;
	devList.push_back(0);

	_1_activateDevices(devList);
	_2_initKernels();
	_3_createBuffer();
	_4_setKernelArgsStatic();
}

std::string OclCompute::readSource(std::string fName) {
	std::ifstream in(fName, std::ios::in | std::ios::binary);
	if (in) {
	    return(std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>()));
	}
	std::cout<<"Couldn't open file:"<<fName<<std::endl;
	return "";
}

std::string OclCompute::getErrorString(cl_int error) {
		switch (error) {
	    	// run-time and JIT compiler errors
	    	case 0: return "CL_SUCCESS";
	    	case -1: return "CL_DEVICE_NOT_FOUND";
	    	case -2: return "CL_DEVICE_NOT_AVAILABLE";
	    	case -3: return "CL_COMPILER_NOT_AVAILABLE";
	    	case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
	    	case -5: return "CL_OUT_OF_RESOURCES";
	    	case -6: return "CL_OUT_OF_HOST_MEMORY";
	    	case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
	    	case -8: return "CL_MEM_COPY_OVERLAP";
	    	case -9: return "CL_IMAGE_FORMAT_MISMATCH";
	    	case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
	    	case -11: return "CL_BUILD_PROGRAM_FAILURE";
	    	case -12: return "CL_MAP_FAILURE";
	    	case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
	    	case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
	    	case -15: return "CL_COMPILE_PROGRAM_FAILURE";
	    	case -16: return "CL_LINKER_NOT_AVAILABLE";
	    	case -17: return "CL_LINK_PROGRAM_FAILURE";
	    	case -18: return "CL_DEVICE_PARTITION_FAILED";
	    	case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

	    	// compile-time errors
	    	case -30: return "CL_INVALID_VALUE";
	    	case -31: return "CL_INVALID_DEVICE_TYPE";
	    	case -32: return "CL_INVALID_PLATFORM";
	    	case -33: return "CL_INVALID_DEVICE";
	    	case -34: return "CL_INVALID_CONTEXT";
	    	case -35: return "CL_INVALID_QUEUE_PROPERTIES";
	    	case -36: return "CL_INVALID_COMMAND_QUEUE";
	    	case -37: return "CL_INVALID_HOST_PTR";
	    	case -38: return "CL_INVALID_MEM_OBJECT";
	    	case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
	    	case -40: return "CL_INVALID_IMAGE_SIZE";
	    	case -41: return "CL_INVALID_SAMPLER";
	    	case -42: return "CL_INVALID_BINARY";
	    	case -43: return "CL_INVALID_BUILD_OPTIONS";
	    	case -44: return "CL_INVALID_PROGRAM";
	    	case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
	    	case -46: return "CL_INVALID_KERNEL_NAME";
	    	case -47: return "CL_INVALID_KERNEL_DEFINITION";
	    	case -48: return "CL_INVALID_KERNEL";
	    	case -49: return "CL_INVALID_ARG_INDEX";
	    	case -50: return "CL_INVALID_ARG_VALUE";
	    	case -51: return "CL_INVALID_ARG_SIZE";
	    	case -52: return "CL_INVALID_KERNEL_ARGS";
	    	case -53: return "CL_INVALID_WORK_DIMENSION";
	    	case -54: return "CL_INVALID_WORK_GROUP_SIZE";
	    	case -55: return "CL_INVALID_WORK_ITEM_SIZE";
	    	case -56: return "CL_INVALID_GLOBAL_OFFSET";
	    	case -57: return "CL_INVALID_EVENT_WAIT_LIST";
	    	case -58: return "CL_INVALID_EVENT";
	    	case -59: return "CL_INVALID_OPERATION";
	    	case -60: return "CL_INVALID_GL_OBJECT";
	    	case -61: return "CL_INVALID_BUFFER_SIZE";
	    	case -62: return "CL_INVALID_MIP_LEVEL";
	    	case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
	    	case -64: return "CL_INVALID_PROPERTY";
	    	case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
	    	case -66: return "CL_INVALID_COMPILER_OPTIONS";
	    	case -67: return "CL_INVALID_LINKER_OPTIONS";
	    	case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

	    	// extension errors
	    	case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
	    	case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
	    	case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
	    	case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
	    	case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
	    	case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
	    	default: return "Unknown OpenCL error";
	    }
	}
