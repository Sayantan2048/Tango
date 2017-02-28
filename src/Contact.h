#ifndef __Contact_h_
#define __Contact_h_

#include <vec3.hpp>
#include <vec4.hpp>
#include <mat4x4.hpp>
#include <mat3x3.hpp>
#include <gtc/quaternion.hpp>
#include <gtx/quaternion.hpp>

#define isnZero(value, threshold) \
        (value <= -threshold || value >= threshold)

#define isZero(value, threshold) \
        (value >= -threshold && value <= threshold)

struct ContactJacobian {
	/* Normal Constraints*/
	glm::dvec3 linN, angN; // Normal constraints
	glm::dvec3 linN_scaledD, angN_scaledD; // Normal constraints scaled by D_row1_inv
	glm::dvec3 linN_scaledM, angN_scaledM; // Normal constraints scaled by inverse mass and inverse inertia

	glm::dvec3 linT1, angT1;
	glm::dvec3 linT2, angT2;
};

class Contact {
	RigidBody *A;
	RigidBody *B;
public:
	ContactJacobian jA;
	ContactJacobian jB;
	double b_row1_scaledD;

	double lambda1;
	double lambda2;
	double lambda3;

	Contact(RigidBody *A, RigidBody *B, const glm::dvec3 &contactPoint, const glm::dvec3 &contactNormal, double dt) {
		A->deltaV = glm::dvec3(0,0,0); A->deltaW = glm::dvec3(0,0,0);
		B->deltaV = glm::dvec3(0,0,0); B->deltaW = glm::dvec3(0,0,0);

		lambda1 = lambda2 = lambda3 = 0;

		this->A = A;
		this->B = B;

		/* compute constraints for Normal direction*/
		jA.linN = -contactNormal; jA.angN = -(A->getRcrossN(contactPoint, contactNormal));
		jB.linN = contactNormal; jB.angN = B->getRcrossN(contactPoint, contactNormal);

		jA.linN_scaledM = A->getScaledByMinv(jA.linN); jA.angN_scaledM = A->getScaledByIinv(jA.angN);
		jB.linN_scaledM = B->getScaledByMinv(jB.linN); jB.angN_scaledM = B->getScaledByIinv(jB.angN);

		double D_row1_inv = glm::dot(jA.linN, jA.linN_scaledM) + glm::dot(jA.angN, jA.angN_scaledM) +
				glm::dot(jB.linN, jB.linN_scaledM) + glm::dot(jB.angN, jB.angN_scaledM);

		if (isZero(D_row1_inv, 1e-6)) {
			std::cerr<<"Two Constrained objects colliding..."<<std::endl;
			exit(0);
		}

		D_row1_inv = 1.0 / D_row1_inv;

		jA.linN_scaledD = jA.linN * D_row1_inv; jA.angN_scaledD = jA.angN * D_row1_inv;
		jB.linN_scaledD = jB.linN * D_row1_inv; jB.angN_scaledD = jB.angN * D_row1_inv;

		glm::dvec3 linImpA = A->getLinearImpulse(dt); glm::dvec3 angImpA = A->getAngularImpulse(dt);
		glm::dvec3 linImpB = B->getLinearImpulse(dt); glm::dvec3 angImpB = B->getAngularImpulse(dt);

		b_row1_scaledD = glm::dot(jA.linN, linImpA) + glm::dot(jA.angN, angImpA) +
				glm::dot(jB.linN, linImpB) + glm::dot(jB.angN, angImpB);
		/*bounce*/// bounce * (c.j1.row1.dot(u1) + c.j2.row1.dot(u2));

		b_row1_scaledD *= D_row1_inv;


		/* Compute constraints for tangential direction 1*/
		glm::dvec3 tangent1;
		if (isnZero(contactNormal.x, 1e-3))
			tangent1 = glm::dvec3(-contactNormal.y-contactNormal.z, contactNormal.x, contactNormal.x);
		else if (isnZero(contactNormal.y, 1e-6))
			tangent1 = glm::dvec3(contactNormal.y, -contactNormal.z - contactNormal.x, contactNormal.y);
		else if (isnZero(contactNormal.z, 1e-16))
			tangent1 = glm::dvec3(contactNormal.z, contactNormal.z, - contactNormal.x - contactNormal.y);
		else {
			std::cerr<<"Contact Normal is zero."<<std::endl;
			exit(0);
		}
		glm::normalize(tangent1);

		jA.linT1 = -tangent1; jA.angT1 = -(A->getRcrossN(contactPoint, tangent1));
		jB.linT1 = tangent1; jB.angT1 = (B->getRcrossN(contactPoint, tangent1));



		/* Compute constraints for tangential direction 2*/
		glm::dvec3 tangent2 = glm::normalize(glm::cross(contactNormal, tangent1));

		jA.linT2 = -tangent2; jA.angT2 = -(A->getRcrossN(contactPoint, tangent2));
		jB.linT2 = tangent2;  jB.angT2 = (B->getRcrossN(contactPoint, tangent2));
	}

	void processContact(double mu) {

	    	double lambda_final1 = lambda1 - b_row1_scaledD - glm::dot(jA.linN_scaledD, A->deltaV)
	    		- glm::dot(jA.angN_scaledD, A->deltaW) - glm::dot(jB.linN_scaledD, B->deltaV)
    		- glm::dot(jB.angN_scaledD, B->deltaW);

	    	if (lambda_final1 < 0) lambda_final1 = 0;

	    	/* double lambda_final2 = c.lambda.y - (c.b_row2_scaledD) - j1.row2_scaled_D.dot(deltaVel[c.body1.index])
	    			- j2.row2_scaled_D.dot(deltaVel[c.body2.index]);

	    	double max_tangent = mu * lambda_final1;
	    	if (lambda_final2 < - max_tangent) lambda_final2 = - max_tangent;
	    	else if (lambda_final2 > max_tangent) lambda_final2 = max_tangent;*/

	    	double delta_lambda1 = lambda_final1 - lambda1;
	    	//double delta_lambda_y = lambda_final2 - c.lambda.y;

	    	A->deltaV += jA.linN_scaledM * delta_lambda1;
	    	A->deltaW += jA.angN_scaledM * delta_lambda1;
	    	/*c.vtemp1.set(j1.row1_scaled_M);
	    	c.vtemp1.scale(delta_lambda_x);
	    	deltaVel[c.body1.index].add(c.vtemp1);
	    	c.vtemp2.set(j1.row2_scaled_M);
	    	c.vtemp2.scale(delta_lambda_y);
	    	deltaVel[c.body1.index].add(c.vtemp2);*/

			B->deltaV += jB.linN_scaledM * delta_lambda1;
	    	B->deltaW += jB.angN_scaledM * delta_lambda1;
	    	/*c.vtemp1.set(j2.row1_scaled_M);
	    	c.vtemp1.scale(delta_lambda_x);
	    	deltaVel[c.body2.index].add(c.vtemp1);
	    	c.vtemp2.set(j2.row2_scaled_M);
	    	c.vtemp2.scale(delta_lambda_y);
	    	deltaVel[c.body2.index].add(c.vtemp2);
	    	c.lambda.set(lambda_final1, lambda_final2);*/

	    	lambda1 = lambda_final1;
	}

	void updateVelocities() {
		A->updateVelocity();
		B->updateVelocity();
	}
};

#endif
