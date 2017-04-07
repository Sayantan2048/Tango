typedef float scalar;
typedef float2 vec2;
typedef uint2 ivec2;

typedef struct {
  vec2 ab;
  scalar c;
} vec3;

typedef struct {
  vec3 vLin;
  vec3 vAng;
} vec6;

__kernel void clearBuffer(__global scalar *deltaVel, __global scalar *lambda, uint nBody, uint nContacts) {
  size_t i = get_global_id(0);
  
  if (i < 6 * nBody)
    deltaVel[i] = 0;
  if (i < 2 * nContacts)
    lambda[i] = 0;
  
  barrier(CLK_GLOBAL_MEM_FENCE);
}

inline scalar dot3(vec3 v1, vec3 v2) {
  return fma(v1.c, v2.c, dot(v1.ab, v2.ab));
}

inline vec3 add3(vec3 v1, vec3 v2) {
  vec3 s;
  s.ab = v1.ab + v2.ab;
  s.c = v1.c + v2.c;
  return s;
}

inline vec3 mul3s(vec3 v, scalar s1) {
  vec3 s;
  s.ab = v.ab * s1;
  s.c = v.c * s1;
  return s;
}

inline void show6(vec6 v) {
  printf("%f %f %f %f %f %f\n", v.vLin.ab.x, v.vLin.ab.y, v.vLin.c, v.vAng.ab.x, v.vAng.ab.y, v.vAng.c);
}

inline vec6 pack6(__global scalar *vIn) {
  vec6 v;
  v.vLin.ab.x = vIn[0];
  v.vLin.ab.y = vIn[1];
  v.vLin.c = vIn[2];
  v.vAng.ab.x = vIn[3];
  v.vAng.ab.y = vIn[4];
  v.vAng.c = vIn[5];
  
  return v;
}

inline ivec2 ipack2(__global uint *vIn) {
  ivec2 v;
  v.x = vIn[0];
  v.y = vIn[1];
  return v;
}

inline vec2 pack2(__global scalar *vIn) {
  vec2 v;
  v.x = vIn[0];
  v.y = vIn[1];
  return v;
}

inline void unpack2(__global scalar *vIn, vec2 v) {
  vIn[0] = v.x;
  vIn[1] = v.y;
}

__kernel void jacobi_parallel(__global scalar *deltaVel, __global uint *bufBodyIndex, __global scalar *bufConstNormalD_A,
	__global scalar *bufConstTangentD_A, __global scalar *bufConstNormalD_B, __global scalar *bufConstTangentD_B, 
	__global scalar *bufB, __global scalar *bufLambda, __global scalar *bufDeltaLambda) {
	
	size_t i = get_global_id(0);
	ivec2 bodyIndex = ipack2(&bufBodyIndex[i<<1]);
	vec6 constNormalD_A = pack6(&bufConstNormalD_A[6 * i]);
	vec6 constNormalD_B = pack6(&bufConstNormalD_B[6 * i]);
	vec6 constTangentD_A = pack6(&bufConstTangentD_A[6 * i]);
	vec6 constTangentD_B = pack6(&bufConstTangentD_B[6 * i]);
	vec2 lambda = pack2(&bufLambda[i<<1]);
	vec2 b = pack2(&bufB[i<<1]);
	vec6 deltaVelA = pack6(&deltaVel[6 * bodyIndex.x]);
	vec6 deltaVelB = pack6(&deltaVel[6 * bodyIndex.y]);
	
	scalar lambda_final1 = lambda.x - b.x - dot3(constNormalD_A.vLin, deltaVelA.vLin)
	    		- dot3(constNormalD_A.vAng, deltaVelA.vAng) - dot3(constNormalD_B.vLin, deltaVelB.vLin)
    			- dot3(constNormalD_B.vAng, deltaVelB.vAng);
	scalar lambda_final2 = lambda.y - b.y - dot3(constTangentD_A.vLin, deltaVelA.vLin)
	    		- dot3(constTangentD_A.vAng, deltaVelA.vAng) - dot3(constTangentD_B.vLin, deltaVelB.vLin)
    			- dot3(constTangentD_B.vAng, deltaVelB.vAng);
	
	//show6(constNormalD_A);
	//printf("%f %f\n", lambda_final1, lambda_final2);		
	lambda_final1 = (lambda_final1 < 0) ? 0 : lambda_final1;
	scalar max_tangent1 = 0.33 * lambda_final1;
	lambda_final2 = (lambda_final2 < -max_tangent1) ? -max_tangent1 : lambda_final2;
	lambda_final2 = (lambda_final2 > max_tangent1) ? max_tangent1 : lambda_final2;
	
	vec2 deltaLambda;
	deltaLambda.x = lambda_final1 - lambda.x;
	deltaLambda.y = lambda_final2 - lambda.y;
	lambda.x = lambda_final1;
	lambda.y = lambda_final2;
	
	unpack2(&bufDeltaLambda[i<<1], deltaLambda);
	unpack2(&bufLambda[i<<1], lambda);
	
	barrier(CLK_GLOBAL_MEM_FENCE);
}

inline void atomicAdd(volatile __global float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

__kernel void jacobi_serial(volatile __global scalar *bufDeltaVel,  __global uint *bufBodyIndex, __global scalar *bufConstNormalM_A,
	__global scalar *bufConstTangentM_A, __global scalar *bufConstNormalM_B, __global scalar *bufConstTangentM_B, 
	__global scalar *bufDeltaLambda) {
	
	size_t i = get_global_id(0);
	ivec2 bodyIndex = ipack2(&bufBodyIndex[i << 1]);
	vec6 constNormalM_A = pack6(&bufConstNormalM_A[6 * i]);
	vec6 constNormalM_B = pack6(&bufConstNormalM_B[6 * i]);
	vec6 constTangentM_A = pack6(&bufConstTangentM_A[6 * i]);
	vec6 constTangentM_B = pack6(&bufConstTangentM_B[6 * i]);
	vec2 deltaLambda = pack2(&bufDeltaLambda[i << 1]);
	
	vec6 deltaVelA;
	vec6 deltaVelB;
  
	deltaVelA.vLin = add3(mul3s(constNormalM_A.vLin, deltaLambda.x), mul3s(constTangentM_A.vLin, deltaLambda.y));
	deltaVelA.vAng = add3(mul3s(constNormalM_A.vAng, deltaLambda.x), mul3s(constTangentM_A.vAng, deltaLambda.y));

	deltaVelB.vLin = add3(mul3s(constNormalM_B.vLin, deltaLambda.x), mul3s(constTangentM_B.vLin, deltaLambda.y));
	deltaVelB.vAng = add3(mul3s(constNormalM_B.vAng, deltaLambda.x), mul3s(constTangentM_B.vAng, deltaLambda.y));

	volatile __global scalar *ptrA = &bufDeltaVel[6 * bodyIndex.x];
	volatile __global scalar *ptrB = &bufDeltaVel[6 * bodyIndex.y];
	
	atomicAdd(&ptrA[0], deltaVelA.vLin.ab.x);
	atomicAdd(&ptrA[1], deltaVelA.vLin.ab.y);
	atomicAdd(&ptrA[2], deltaVelA.vLin.c);
	atomicAdd(&ptrA[3], deltaVelA.vAng.ab.x);
	atomicAdd(&ptrA[4], deltaVelA.vAng.ab.y);
	atomicAdd(&ptrA[5], deltaVelA.vAng.c);
	
	atomicAdd(&ptrB[0], deltaVelB.vLin.ab.x);
	atomicAdd(&ptrB[1], deltaVelB.vLin.ab.y);
	atomicAdd(&ptrB[2], deltaVelB.vLin.c);
	atomicAdd(&ptrB[3], deltaVelB.vAng.ab.x);
	atomicAdd(&ptrB[4], deltaVelB.vAng.ab.y);
	atomicAdd(&ptrB[5], deltaVelB.vAng.c);
	
	barrier(CLK_GLOBAL_MEM_FENCE);
}