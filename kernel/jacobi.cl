typedef float scalar;
typedef float2 vec2;
typedef float4 vec4;
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

inline scalar dot6(vec6 v1, vec6 v2) {
  return dot3(v1.vLin, v2.vLin) + dot3(v1.vAng, v2.vAng);
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

inline void ishow2(ivec2 v) {
  printf("%d %d \n", v.x, v.y);
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

inline vec6 s_pack6(__local scalar *vIn) {
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

inline ivec2 s_ipack2(__local uint *vIn) {
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

//http://suhorukov.blogspot.ca/2011/12/opencl-11-atomic-operations-on-floating.html
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
  // Loop until source stops changing   
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

__kernel void jacobi_comb(volatile __global scalar *deltaVel, __global uint *bufBodyIndex, __global scalar *bufConstNormalD_A,
	__global scalar *bufConstTangentD_A, __global scalar *bufConstNormalD_B, __global scalar *bufConstTangentD_B,
	__global scalar *bufConstNormalM_A, __global scalar *bufConstTangentM_A, __global scalar *bufConstNormalM_B, 
	__global scalar *bufConstTangentM_B, __global scalar *bufB)
{
  size_t i = get_global_id(0);
  ivec2 bodyIndex = ipack2(&bufBodyIndex[i<<1]);
  vec6 constNormalD_A = pack6(&bufConstNormalD_A[6 * i]);
  vec6 constNormalD_B = pack6(&bufConstNormalD_B[6 * i]);
  vec6 constTangentD_A = pack6(&bufConstTangentD_A[6 * i]);
  vec6 constTangentD_B = pack6(&bufConstTangentD_B[6 * i]);
  vec6 constNormalM_A = pack6(&bufConstNormalM_A[6 * i]);
  vec6 constNormalM_B = pack6(&bufConstNormalM_B[6 * i]);
  vec6 constTangentM_A = pack6(&bufConstTangentM_A[6 * i]);
  vec6 constTangentM_B = pack6(&bufConstTangentM_B[6 * i]);
  vec2 b = pack2(&bufB[i<<1]);

  scalar lambda1;
  scalar lambda2;
  lambda1 = 0;
  lambda2 = 0;
  
  int iter;
  
  volatile __global scalar *ptrA = &deltaVel[6 * bodyIndex.x];
  volatile __global scalar *ptrB = &deltaVel[6 * bodyIndex.y];
  
  for (iter = 0; iter < 250; iter++) {
	vec6 deltaVelA = pack6(ptrA);
	vec6 deltaVelB = pack6(ptrB);
	
	scalar lambda_final1 = lambda1 - b.x - dot3(constNormalD_A.vLin, deltaVelA.vLin)
	    		- dot3(constNormalD_A.vAng, deltaVelA.vAng) - dot3(constNormalD_B.vLin, deltaVelB.vLin)
    			- dot3(constNormalD_B.vAng, deltaVelB.vAng);
	scalar lambda_final2 = lambda2 - b.y - dot3(constTangentD_A.vLin, deltaVelA.vLin)
	    		- dot3(constTangentD_A.vAng, deltaVelA.vAng) - dot3(constTangentD_B.vLin, deltaVelB.vLin)
    			- dot3(constTangentD_B.vAng, deltaVelB.vAng);
	
	//show6(constNormalD_A);
	//printf("%f %f\n", lambda_final1, lambda_final2);		
	lambda_final1 = (lambda_final1 < 0) ? 0 : lambda_final1;
	scalar max_tangent1 = 0.33 * lambda_final1;
	lambda_final2 = (lambda_final2 < -max_tangent1) ? -max_tangent1 : lambda_final2;
	lambda_final2 = (lambda_final2 > max_tangent1) ? max_tangent1 : lambda_final2;
	
	scalar deltaLambda1;
	scalar deltaLambda2;
	deltaLambda1 = lambda_final1 - lambda1;
	deltaLambda2 = lambda_final2 - lambda2;
	lambda1 = lambda_final1;
	lambda2 = lambda_final2;
	
	deltaVelA.vLin = add3(mul3s(constNormalM_A.vLin, deltaLambda1), mul3s(constTangentM_A.vLin, deltaLambda2));
	deltaVelA.vAng = add3(mul3s(constNormalM_A.vAng, deltaLambda1), mul3s(constTangentM_A.vAng, deltaLambda2));

	deltaVelB.vLin = add3(mul3s(constNormalM_B.vLin, deltaLambda1), mul3s(constTangentM_B.vLin, deltaLambda2));
	deltaVelB.vAng = add3(mul3s(constNormalM_B.vAng, deltaLambda1), mul3s(constTangentM_B.vAng, deltaLambda2));
	
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
  
}
/*
  a1 = 0;
  a2 = 0;
  for (j = 0; j < numContacts; j++) {
    bodyIndexI = ipack2(&bufBodyIndex[j<<1]);
    if (bodyIndex.x == bodyIndexI.x){
      vec2 lambda = pack2(&bufLambda[j<<1]);
      
      vec6 cn = pack6(&bufConstNormalM_A[6 * j]);
      vec6 ct = pack6(&bufConstTangentM_A[6 * j]);
      a1 += dot6(constNormalD_A, cn) * lambda.x + dot6(constNormalD_A, ct) * lambda.y;
      a2 += dot6(constTangentD_A, cn) * lambda.x + dot6(constTangentD_A, ct) * lambda.y;
    }
    if (bodyIndex.x == bodyIndexI.y){
      vec2 lambda = pack2(&bufLambda[j<<1]);
      
      vec6 cn = pack6(&bufConstNormalM_B[6 * j]);
      vec6 ct = pack6(&bufConstTangentM_B[6 * j]);
      a1 += dot6(constNormalD_A, cn) * lambda.x + dot6(constNormalD_A, ct) * lambda.y;
      a2 += dot6(constTangentD_A, cn) * lambda.x + dot6(constTangentD_A, ct) * lambda.y;
    }
    if (bodyIndex.y == bodyIndexI.x){
      vec2 lambda = pack2(&bufLambda[j<<1]);
      
      vec6 cn = pack6(&bufConstNormalM_A[6 * j]);
      vec6 ct = pack6(&bufConstTangentM_A[6 * j]);
      a1 += dot6(constNormalD_B, cn) * lambda.x + dot6(constNormalD_B, ct) * lambda.y;
      a2 += dot6(constTangentD_B, cn) * lambda.x + dot6(constTangentD_B, ct) * lambda.y;
    }
    if (bodyIndex.y == bodyIndexI.y){
      vec2 lambda = pack2(&bufLambda[j<<1]);
      vec6 cn = pack6(&bufConstNormalM_B[6 * j]);
      vec6 ct = pack6(&bufConstTangentM_B[6 * j]);
      a1 += dot6(constNormalD_B, cn) * lambda.x + dot6(constNormalD_B, ct) * lambda.y;
      a2 += dot6(constTangentD_B, cn) * lambda.x + dot6(constTangentD_B, ct) * lambda.y;
    }
  }


*/
__kernel void jacobi_v2(volatile __global scalar *deltaVel, __global uint *bufBodyIndex, __global scalar *bufConstNormalD_A,
	__global scalar *bufConstTangentD_A, __global scalar *bufConstNormalD_B, __global scalar *bufConstTangentD_B,
	__global scalar *bufConstNormalM_A, __global scalar *bufConstTangentM_A, __global scalar *bufConstNormalM_B, 
	__global scalar *bufConstTangentM_B, __global scalar *bufB, __global scalar *bufLambda, uint numContacts,
	__local uint *s_bufBodyIndex, __local scalar *s_bufConstNormalM_A, __local scalar *s_bufConstTangentM_A,
	__local scalar *s_bufConstNormalM_B, __local scalar *s_bufConstTangentM_B)
{
  size_t i = get_global_id(0);
  ivec2 bodyIndex = ipack2(&bufBodyIndex[i<<1]);
  vec6 constNormalD_A = pack6(&bufConstNormalD_A[6 * i]);
  vec6 constNormalD_B = pack6(&bufConstNormalD_B[6 * i]);
  vec6 constTangentD_A = pack6(&bufConstTangentD_A[6 * i]);
  vec6 constTangentD_B = pack6(&bufConstTangentD_B[6 * i]);
  vec6 constNormalM_A = pack6(&bufConstNormalM_A[6 * i]);
  vec6 constNormalM_B = pack6(&bufConstNormalM_B[6 * i]);
  vec6 constTangentM_A = pack6(&bufConstTangentM_A[6 * i]);
  vec6 constTangentM_B = pack6(&bufConstTangentM_B[6 * i]);
  vec2 b = pack2(&bufB[i<<1]);
  vec2 lambda;
  
  lambda.x = 0;
  lambda.y = 0;
  uint iter;
  uint s;
  
  for (s = get_local_id(0); s < numContacts; s += get_local_size(0)) {
    uint temp = s << 1;
    s_bufBodyIndex[temp] = bufBodyIndex[temp];
    s_bufBodyIndex[temp + 1] = bufBodyIndex[temp + 1];
    temp = s * 6;
    s_bufConstNormalM_A[temp] = bufConstNormalM_A[temp];
    s_bufConstNormalM_A[temp + 1] = bufConstNormalM_A[temp + 1];
    s_bufConstNormalM_A[temp + 2] = bufConstNormalM_A[temp + 2];
    s_bufConstNormalM_A[temp + 3] = bufConstNormalM_A[temp + 3];
    s_bufConstNormalM_A[temp + 4] = bufConstNormalM_A[temp + 4];
    s_bufConstNormalM_A[temp + 5] = bufConstNormalM_A[temp + 5];
    
    s_bufConstTangentM_A[temp] = bufConstTangentM_A[temp];
    s_bufConstTangentM_A[temp + 1] = bufConstTangentM_A[temp + 1];
    s_bufConstTangentM_A[temp + 2] = bufConstTangentM_A[temp + 2];
    s_bufConstTangentM_A[temp + 3] = bufConstTangentM_A[temp + 3];
    s_bufConstTangentM_A[temp + 4] = bufConstTangentM_A[temp + 4];
    s_bufConstTangentM_A[temp + 5] = bufConstTangentM_A[temp + 5];
    
    s_bufConstNormalM_B[temp] = bufConstNormalM_B[temp];
    s_bufConstNormalM_B[temp + 1] = bufConstNormalM_B[temp + 1];
    s_bufConstNormalM_B[temp + 2] = bufConstNormalM_B[temp + 2];
    s_bufConstNormalM_B[temp + 3] = bufConstNormalM_B[temp + 3];
    s_bufConstNormalM_B[temp + 4] = bufConstNormalM_B[temp + 4];
    s_bufConstNormalM_B[temp + 5] = bufConstNormalM_B[temp + 5];
    
    s_bufConstTangentM_B[temp] = bufConstTangentM_B[temp];
    s_bufConstTangentM_B[temp + 1] = bufConstTangentM_B[temp + 1];
    s_bufConstTangentM_B[temp + 2] = bufConstTangentM_B[temp + 2];
    s_bufConstTangentM_B[temp + 3] = bufConstTangentM_B[temp + 3];
    s_bufConstTangentM_B[temp + 4] = bufConstTangentM_B[temp + 4];
    s_bufConstTangentM_B[temp + 5] = bufConstTangentM_B[temp + 5];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
    
  volatile __global scalar *ptrA = &deltaVel[6 * bodyIndex.x];
  volatile __global scalar *ptrB = &deltaVel[6 * bodyIndex.y];
  
  for (iter = 0; iter < 500; iter++) {
    scalar a1 = 0;
    scalar a2 = 0;
    uint j;
    for (j = 0; j < numContacts; j++) {
      ivec2 bodyIndexI = s_ipack2(&s_bufBodyIndex[j<<1]);
      vec2 lambdaI;
      if (bodyIndex.x == bodyIndexI.x || bodyIndex.x == bodyIndexI.y || bodyIndex.y == bodyIndexI.x ||  bodyIndex.y == bodyIndexI.y){
	lambdaI = pack2(&bufLambda[j<<1]);
	//vec6 cnA = pack6(&bufConstNormalM_A[6 * j]);
      }
      if (bodyIndex.x == bodyIndexI.x){
	vec6 cnA = s_pack6(&s_bufConstNormalM_A[6 * j]);
	vec6 ctA = s_pack6(&s_bufConstTangentM_A[6 * j]);
	a1 += dot6(constNormalD_A, cnA) * lambdaI.x + dot6(constNormalD_A, ctA) * lambdaI.y;
	a2 += dot6(constTangentD_A, cnA) * lambdaI.x + dot6(constTangentD_A, ctA) * lambdaI.y;
      }
      else if (bodyIndex.x == bodyIndexI.y){
	//vec2 lambdaI = pack2(&bufLambda[j<<1]);
      
	vec6 cnB = s_pack6(&s_bufConstNormalM_B[6 * j]);
	vec6 ctB = s_pack6(&s_bufConstTangentM_B[6 * j]);
	a1 += dot6(constNormalD_A, cnB) * lambdaI.x + dot6(constNormalD_A, ctB) * lambdaI.y;
	a2 += dot6(constTangentD_A, cnB) * lambdaI.x + dot6(constTangentD_A, ctB) * lambdaI.y;
      }
      if (bodyIndex.y == bodyIndexI.x){
	//vec2 lambdaI = pack2(&bufLambda[j<<1]);
      
	vec6 cnA = s_pack6(&s_bufConstNormalM_A[6 * j]);
	vec6 ctA = s_pack6(&s_bufConstTangentM_A[6 * j]);
	a1 += dot6(constNormalD_B, cnA) * lambdaI.x + dot6(constNormalD_B, ctA) * lambdaI.y;
	a2 += dot6(constTangentD_B, cnA) * lambdaI.x + dot6(constTangentD_B, ctA) * lambdaI.y;
      }
      else if (bodyIndex.y == bodyIndexI.y){
	//vec2 lambdaI = pack2(&bufLambda[j<<1]);
	vec6 cnB = s_pack6(&s_bufConstNormalM_B[6 * j]);
	vec6 ctB = s_pack6(&s_bufConstTangentM_B[6 * j]);
	a1 += dot6(constNormalD_B, cnB) * lambdaI.x + dot6(constNormalD_B, ctB) * lambdaI.y;
	a2 += dot6(constTangentD_B, cnB) * lambdaI.x + dot6(constTangentD_B, ctB) * lambdaI.y;
      }
    }
    
    lambda.x = lambda.x - b.x - a1;
    lambda.y = lambda.y - b.y - a2;
    
    lambda.x = (lambda.x < 0) ? 0 : lambda.x;
    scalar max_tangent1 = 0.33 * lambda.x;
    lambda.y = (lambda.y < -max_tangent1) ? -max_tangent1 : lambda.y;
    lambda.y = (lambda.y > max_tangent1) ? max_tangent1 : lambda.y;
        
    unpack2(&bufLambda[i<<1], lambda);
    
    barrier(CLK_GLOBAL_MEM_FENCE);
  }
  
  vec6 deltaVelA;
  vec6 deltaVelB;
  
  deltaVelA.vLin = add3(mul3s(constNormalM_A.vLin, lambda.x), mul3s(constTangentM_A.vLin, lambda.y));
  deltaVelA.vAng = add3(mul3s(constNormalM_A.vAng, lambda.x), mul3s(constTangentM_A.vAng, lambda.y));

  deltaVelB.vLin = add3(mul3s(constNormalM_B.vLin, lambda.x), mul3s(constTangentM_B.vLin, lambda.y));
  deltaVelB.vAng = add3(mul3s(constNormalM_B.vAng, lambda.x), mul3s(constTangentM_B.vAng, lambda.y));
	
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
}

__kernel void jacobi_v3(volatile __global scalar *deltaVel, __global ivec2 *bufBodyIndex, __global scalar *bufConstNormalD_A,
	__global scalar *bufConstTangentD_A, __global scalar *bufConstNormalD_B, __global scalar *bufConstTangentD_B,
	__global scalar *bufConstNormalM_A, __global scalar *bufConstTangentM_A, __global scalar *bufConstNormalM_B, 
	__global scalar *bufConstTangentM_B, __global vec2 *bufB, __global vec2 *bufLambda, __global vec4 *matrixA, uint numContacts,
	__local ivec2 *s_bufBodyIndex, __local scalar *s_bufConstNormalM_A, __local scalar *s_bufConstTangentM_A,
	__local scalar *s_bufConstNormalM_B, __local scalar *s_bufConstTangentM_B)
{
  size_t i = get_global_id(0);
  ivec2 bodyIndex = bufBodyIndex[i];
 {
    vec6 constNormalD_A = pack6(&bufConstNormalD_A[6 * i]);
    vec6 constNormalD_B = pack6(&bufConstNormalD_B[6 * i]);
    vec6 constTangentD_A = pack6(&bufConstTangentD_A[6 * i]);
    vec6 constTangentD_B = pack6(&bufConstTangentD_B[6 * i]);
 
    uint s;
    for (s = get_local_id(0); s < numContacts; s += get_local_size(0)) {
	s_bufBodyIndex[s] = bufBodyIndex[s];
   
	uint temp = s * 6;
	s_bufConstNormalM_A[temp] = bufConstNormalM_A[temp];
	s_bufConstNormalM_A[temp + 1] = bufConstNormalM_A[temp + 1];
	s_bufConstNormalM_A[temp + 2] = bufConstNormalM_A[temp + 2];
	s_bufConstNormalM_A[temp + 3] = bufConstNormalM_A[temp + 3];
	s_bufConstNormalM_A[temp + 4] = bufConstNormalM_A[temp + 4];
	s_bufConstNormalM_A[temp + 5] = bufConstNormalM_A[temp + 5];
    
	s_bufConstTangentM_A[temp] = bufConstTangentM_A[temp];
	s_bufConstTangentM_A[temp + 1] = bufConstTangentM_A[temp + 1];
	s_bufConstTangentM_A[temp + 2] = bufConstTangentM_A[temp + 2];
	s_bufConstTangentM_A[temp + 3] = bufConstTangentM_A[temp + 3];
	s_bufConstTangentM_A[temp + 4] = bufConstTangentM_A[temp + 4];
	s_bufConstTangentM_A[temp + 5] = bufConstTangentM_A[temp + 5];
    
	s_bufConstNormalM_B[temp] = bufConstNormalM_B[temp];
	s_bufConstNormalM_B[temp + 1] = bufConstNormalM_B[temp + 1];
	s_bufConstNormalM_B[temp + 2] = bufConstNormalM_B[temp + 2];
	s_bufConstNormalM_B[temp + 3] = bufConstNormalM_B[temp + 3];
	s_bufConstNormalM_B[temp + 4] = bufConstNormalM_B[temp + 4];
	s_bufConstNormalM_B[temp + 5] = bufConstNormalM_B[temp + 5];
    
	s_bufConstTangentM_B[temp] = bufConstTangentM_B[temp];
	s_bufConstTangentM_B[temp + 1] = bufConstTangentM_B[temp + 1];
	s_bufConstTangentM_B[temp + 2] = bufConstTangentM_B[temp + 2];
	s_bufConstTangentM_B[temp + 3] = bufConstTangentM_B[temp + 3];
	s_bufConstTangentM_B[temp + 4] = bufConstTangentM_B[temp + 4];
	s_bufConstTangentM_B[temp + 5] = bufConstTangentM_B[temp + 5];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  
    for (s = 0; s < numContacts; s++) {
      ivec2 bodyIndexI = s_bufBodyIndex[s];
      uint temp = s * 6;
      vec4 a;
      a.x = 0, a.y = 0, a.z = 0, a.w = 0;
      if (bodyIndex.x == bodyIndexI.x) {
	  vec6 cnA = s_pack6(&s_bufConstNormalM_A[temp]);
	  vec6 ctA = s_pack6(&s_bufConstTangentM_A[temp]);
	  a.x = dot6(constNormalD_A, cnA); a.y = dot6(constNormalD_A, ctA);
	  a.z = dot6(constTangentD_A, cnA); a.w = dot6(constTangentD_A, ctA);
      }
      else if (bodyIndex.x == bodyIndexI.y) {
	  vec6 cnB = s_pack6(&s_bufConstNormalM_B[temp]);
	  vec6 ctB = s_pack6(&s_bufConstTangentM_B[temp]);
	  a.x = dot6(constNormalD_A, cnB); a.y = dot6(constNormalD_A, ctB);
	  a.z = dot6(constTangentD_A, cnB); a.w = dot6(constTangentD_A, ctB);
      }
      if (bodyIndex.y == bodyIndexI.x) {
	  vec6 cnA = s_pack6(&s_bufConstNormalM_A[temp]);
	  vec6 ctA = s_pack6(&s_bufConstTangentM_A[temp]);
	  a.x += dot6(constNormalD_B, cnA); a.y += dot6(constNormalD_B, ctA);
	  a.z += dot6(constTangentD_B, cnA); a.w += dot6(constTangentD_B, ctA);
      }
      else if (bodyIndex.y == bodyIndexI.y) {
	  vec6 cnB = s_pack6(&s_bufConstNormalM_B[temp]);
	  vec6 ctB = s_pack6(&s_bufConstTangentM_B[temp]);
	  a.x += dot6(constNormalD_B, cnB); a.y += dot6(constNormalD_B, ctB);
	  a.z += dot6(constTangentD_B, cnB); a.w += dot6(constTangentD_B, ctB);
      }
       
      matrixA[s * numContacts + i] = a;
    }
  }
  
  barrier(CLK_GLOBAL_MEM_FENCE);
  vec2 lambda;
  lambda.x = 0;
  lambda.y = 0;
  
  {
    vec2 b = bufB[i];
    uint iter;
  
    for (iter = 0; iter < 500; iter++) {
      scalar a1 = 0;
      scalar a2 = 0;
      uint j;
      for (j = 0; j < numContacts; j++) {
	vec4 a =  matrixA[j * numContacts + i];
	vec2 lambdaI = bufLambda[j];
	a1 += a.x * lambdaI.x + a.y * lambdaI.y;
	a2 += a.z * lambdaI.x + a.w * lambdaI.y;
      }
    
      lambda.x = lambda.x - b.x - a1;
      lambda.y = lambda.y - b.y - a2;
    
      lambda.x = (lambda.x < 0) ? 0 : lambda.x;
      scalar max_tangent1 = 0.33 * lambda.x;
      lambda.y = (lambda.y < -max_tangent1) ? -max_tangent1 : lambda.y;
      lambda.y = (lambda.y > max_tangent1) ? max_tangent1 : lambda.y;
        
      bufLambda[i] = lambda;
    
      barrier(CLK_GLOBAL_MEM_FENCE);
    }
  }
  
  {
    volatile __global scalar *ptrA = &deltaVel[6 * bodyIndex.x];
    volatile __global scalar *ptrB = &deltaVel[6 * bodyIndex.y];
  
    vec6 constNormalM_A = pack6(&bufConstNormalM_A[6 * i]);
    vec6 constNormalM_B = pack6(&bufConstNormalM_B[6 * i]);
    vec6 constTangentM_A = pack6(&bufConstTangentM_A[6 * i]);
    vec6 constTangentM_B = pack6(&bufConstTangentM_B[6 * i]);
  
    vec6 deltaVelA;
    vec6 deltaVelB;
  
    deltaVelA.vLin = add3(mul3s(constNormalM_A.vLin, lambda.x), mul3s(constTangentM_A.vLin, lambda.y));
    deltaVelA.vAng = add3(mul3s(constNormalM_A.vAng, lambda.x), mul3s(constTangentM_A.vAng, lambda.y));

    deltaVelB.vLin = add3(mul3s(constNormalM_B.vLin, lambda.x), mul3s(constTangentM_B.vLin, lambda.y));
    deltaVelB.vAng = add3(mul3s(constNormalM_B.vAng, lambda.x), mul3s(constTangentM_B.vAng, lambda.y));
	
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
  }
}

__kernel void jacobi_v3_split1(__global ivec2 *bufBodyIndex, __global scalar *bufConstNormalD_A,
	__global scalar *bufConstTangentD_A, __global scalar *bufConstNormalD_B, __global scalar *bufConstTangentD_B,
	__global scalar *bufConstNormalM_A, __global scalar *bufConstTangentM_A, __global scalar *bufConstNormalM_B, 
	__global scalar *bufConstTangentM_B, __global vec4 *matrixA, uint numContacts,
	__local ivec2 *s_bufBodyIndex, __local scalar *s_bufConstNormalM_A, __local scalar *s_bufConstTangentM_A,
	__local scalar *s_bufConstNormalM_B, __local scalar *s_bufConstTangentM_B)
{
  size_t i = get_global_id(0);
  ivec2 bodyIndex = bufBodyIndex[i];
 {
    vec6 constNormalD_A = pack6(&bufConstNormalD_A[6 * i]);
    vec6 constNormalD_B = pack6(&bufConstNormalD_B[6 * i]);
    vec6 constTangentD_A = pack6(&bufConstTangentD_A[6 * i]);
    vec6 constTangentD_B = pack6(&bufConstTangentD_B[6 * i]);
 
    uint s;
    for (s = get_local_id(0); s < numContacts; s += get_local_size(0)) {
	s_bufBodyIndex[s] = bufBodyIndex[s];
   
	uint temp = s * 6;
	s_bufConstNormalM_A[temp] = bufConstNormalM_A[temp];
	s_bufConstNormalM_A[temp + 1] = bufConstNormalM_A[temp + 1];
	s_bufConstNormalM_A[temp + 2] = bufConstNormalM_A[temp + 2];
	s_bufConstNormalM_A[temp + 3] = bufConstNormalM_A[temp + 3];
	s_bufConstNormalM_A[temp + 4] = bufConstNormalM_A[temp + 4];
	s_bufConstNormalM_A[temp + 5] = bufConstNormalM_A[temp + 5];
    
	s_bufConstTangentM_A[temp] = bufConstTangentM_A[temp];
	s_bufConstTangentM_A[temp + 1] = bufConstTangentM_A[temp + 1];
	s_bufConstTangentM_A[temp + 2] = bufConstTangentM_A[temp + 2];
	s_bufConstTangentM_A[temp + 3] = bufConstTangentM_A[temp + 3];
	s_bufConstTangentM_A[temp + 4] = bufConstTangentM_A[temp + 4];
	s_bufConstTangentM_A[temp + 5] = bufConstTangentM_A[temp + 5];
    
	s_bufConstNormalM_B[temp] = bufConstNormalM_B[temp];
	s_bufConstNormalM_B[temp + 1] = bufConstNormalM_B[temp + 1];
	s_bufConstNormalM_B[temp + 2] = bufConstNormalM_B[temp + 2];
	s_bufConstNormalM_B[temp + 3] = bufConstNormalM_B[temp + 3];
	s_bufConstNormalM_B[temp + 4] = bufConstNormalM_B[temp + 4];
	s_bufConstNormalM_B[temp + 5] = bufConstNormalM_B[temp + 5];
    
	s_bufConstTangentM_B[temp] = bufConstTangentM_B[temp];
	s_bufConstTangentM_B[temp + 1] = bufConstTangentM_B[temp + 1];
	s_bufConstTangentM_B[temp + 2] = bufConstTangentM_B[temp + 2];
	s_bufConstTangentM_B[temp + 3] = bufConstTangentM_B[temp + 3];
	s_bufConstTangentM_B[temp + 4] = bufConstTangentM_B[temp + 4];
	s_bufConstTangentM_B[temp + 5] = bufConstTangentM_B[temp + 5];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  
    for (s = 0; s < numContacts; s++) {
      ivec2 bodyIndexI = s_bufBodyIndex[s];
      uint temp = s * 6;
      vec4 a;
      a.x = 0, a.y = 0, a.z = 0, a.w = 0;
      if (bodyIndex.x == bodyIndexI.x) {
	  vec6 cnA = s_pack6(&s_bufConstNormalM_A[temp]);
	  vec6 ctA = s_pack6(&s_bufConstTangentM_A[temp]);
	  a.x = dot6(constNormalD_A, cnA); a.y = dot6(constNormalD_A, ctA);
	  a.z = dot6(constTangentD_A, cnA); a.w = dot6(constTangentD_A, ctA);
      }
      else if (bodyIndex.x == bodyIndexI.y) {
	  vec6 cnB = s_pack6(&s_bufConstNormalM_B[temp]);
	  vec6 ctB = s_pack6(&s_bufConstTangentM_B[temp]);
	  a.x = dot6(constNormalD_A, cnB); a.y = dot6(constNormalD_A, ctB);
	  a.z = dot6(constTangentD_A, cnB); a.w = dot6(constTangentD_A, ctB);
      }
      if (bodyIndex.y == bodyIndexI.x) {
	  vec6 cnA = s_pack6(&s_bufConstNormalM_A[temp]);
	  vec6 ctA = s_pack6(&s_bufConstTangentM_A[temp]);
	  a.x += dot6(constNormalD_B, cnA); a.y += dot6(constNormalD_B, ctA);
	  a.z += dot6(constTangentD_B, cnA); a.w += dot6(constTangentD_B, ctA);
      }
      else if (bodyIndex.y == bodyIndexI.y) {
	  vec6 cnB = s_pack6(&s_bufConstNormalM_B[temp]);
	  vec6 ctB = s_pack6(&s_bufConstTangentM_B[temp]);
	  a.x += dot6(constNormalD_B, cnB); a.y += dot6(constNormalD_B, ctB);
	  a.z += dot6(constTangentD_B, cnB); a.w += dot6(constTangentD_B, ctB);
      }
       
      matrixA[i * numContacts + s] = a;
    }
  }
}
__kernel void jacobi_v3_split2(volatile __global scalar *deltaVel, __global ivec2 *bufBodyIndex, __global scalar *bufConstNormalM_A, 
	__global scalar *bufConstTangentM_A, __global scalar *bufConstNormalM_B, __global scalar *bufConstTangentM_B, 
	__global vec2 *bufB, __global vec2 *bufLambda, __global vec4 *matrixA, uint numContacts) {
  
  size_t i = get_global_id(0);
  ivec2 bodyIndex = bufBodyIndex[i];
  
  vec2 lambda;
  lambda.x = 0;
  lambda.y = 0;
  
  {
    vec2 b = bufB[i];
    uint iter;
  
    for (iter = 0; iter < 250; iter++) {
      scalar a1 = 0;
      scalar a2 = 0;
      uint j;
      for (j = 0; j < numContacts; j++) {
	vec4 a =  matrixA[i * numContacts + j];
	vec2 lambdaI = bufLambda[j];
	a1 += a.x * lambdaI.x + a.y * lambdaI.y;
	a2 += a.z * lambdaI.x + a.w * lambdaI.y;
      }
    
      lambda.x = lambda.x - b.x - a1;
      lambda.y = lambda.y - b.y - a2;
    
      lambda.x = (lambda.x < 0) ? 0 : lambda.x;
      scalar max_tangent1 = 0.33 * lambda.x;
      lambda.y = (lambda.y < -max_tangent1) ? -max_tangent1 : lambda.y;
      lambda.y = (lambda.y > max_tangent1) ? max_tangent1 : lambda.y;
        
      barrier(CLK_GLOBAL_MEM_FENCE);
      bufLambda[i] = lambda;
      barrier(CLK_GLOBAL_MEM_FENCE);
      
    }
  }
  
  {
    volatile __global scalar *ptrA = &deltaVel[6 * bodyIndex.x];
    volatile __global scalar *ptrB = &deltaVel[6 * bodyIndex.y];
  
    vec6 constNormalM_A = pack6(&bufConstNormalM_A[6 * i]);
    vec6 constNormalM_B = pack6(&bufConstNormalM_B[6 * i]);
    vec6 constTangentM_A = pack6(&bufConstTangentM_A[6 * i]);
    vec6 constTangentM_B = pack6(&bufConstTangentM_B[6 * i]);
  
    vec6 deltaVelA;
    vec6 deltaVelB;
  
    deltaVelA.vLin = add3(mul3s(constNormalM_A.vLin, lambda.x), mul3s(constTangentM_A.vLin, lambda.y));
    deltaVelA.vAng = add3(mul3s(constNormalM_A.vAng, lambda.x), mul3s(constTangentM_A.vAng, lambda.y));

    deltaVelB.vLin = add3(mul3s(constNormalM_B.vLin, lambda.x), mul3s(constTangentM_B.vLin, lambda.y));
    deltaVelB.vAng = add3(mul3s(constNormalM_B.vAng, lambda.x), mul3s(constTangentM_B.vAng, lambda.y));
	
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
  }
}

