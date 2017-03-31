#ifndef __RigidBody_h_
#define __RigidBody_h_

#include <vec3.hpp>
#include <vec4.hpp>
#include <mat4x4.hpp>
#include <mat3x3.hpp>
#include <gtc/quaternion.hpp>
#include <gtx/quaternion.hpp>
#include <gtc/matrix_access.hpp>
#include "BaseApplication.h"
#include <btBulletDynamicsCommon.h>
#include "BulletCollision/CollisionShapes/btShapeHull.h"

class RigidBody {
	size_t vertex_count;
	size_t index_count;

	unsigned long *indices; // Do not change type, memory alloc issue
	glm::dvec3 *vertices; // Do not change type, memory alloc issue

	/* Linear Velocity */
	glm::dvec3 v;
	/* Angular Velocity */
	glm::dvec3 w;

	/* Force acting on the body*/
	glm::dvec3 f;
	/* Torque acting on the body*/
	glm::dvec3 t;

	/* Position of CM */
	glm::dvec3 p;
	/* Body to World */
	glm::dquat b2w_rot;
	/* World to Body */
	glm::dmat4x4 w2b;
	/* Body to World*/
	glm::dmat4x4 b2w;

	/* Inertia Tensor at default orientation */
	glm::dmat3x3 iT_0;
	/* Inertia Tensor at given orientation */
	glm::dmat3x3 iT;

	/* Inverse Inertia Tensor at default orientation */
	glm::dmat3x3 iiT_0;

	/* Inverse Inertia Tensor at given orientation */
	glm::dmat3x3 iiT;

	double iMass;

	/* Body is constrained */
	bool constrained;

	Ogre::SceneNode *node;

	/* Bullet Collision Object */
	btCollisionObject* collisionObject;
	btConvexHullShape* simplifiedConvexShape;

	/* Passed from main program, do not delete!!*/
	btCollisionWorld *collisionWorld;

	/* Must be called after initializing collisionObject */
	void updateTransform();

public:
	unsigned long int index;
	/* For contact processing */
	glm::dvec3 deltaV; // delta linear velocity
	glm::dvec3 deltaW; // delta angular velocity

	void advanceTime(double dt);
	void applyForce(glm::dvec3 contact, glm::dvec3 force);
	void applyForce(glm::dvec3 force){f += force;};

	inline glm::dvec3 getRcrossN(const glm::dvec3 &contact, const glm::dvec3 &normal) const { return glm::cross(contact - p, normal);}
	//Scale a vector by inverse mass
	inline glm::dvec3 getScaledByMinv(const glm::dvec3 &vec) const {return vec * iMass;}
	//Multiply a vector by inverse inertia tensor
	inline glm::dvec3 getScaledByIinv(const glm::dvec3 &vec) const {return iiT * vec;}
	inline glm::dvec3 getLinearImpulse(double dt) const {return v + f * (dt * iMass);}
	inline glm::dvec3 getAngularImpulse(double dt) const {return w + (constrained ? glm::dvec3(0,0,0) : iiT * ((t - glm::cross(w, iT * w)) * dt));}
	inline double getDotWithV(const glm::dvec3 &vec) const { return glm::dot(v, vec);}
	inline double getDotWithW(const glm::dvec3 &vec) const { return glm::dot(w, vec);}
	inline void updateVelocity() { v += deltaV; w += deltaW; deltaV = glm::dvec3(0,0,0); deltaW = glm::dvec3(0,0,0);}

	glm::dvec4 getWorldToBody(glm::dvec4 world);
	glm::dvec4 getBodyToWorld(glm::dvec4 body);
	glm::dvec3 getContactVelocity(glm::dvec3 contactPoint);

	RigidBody(unsigned long int index, Ogre::Entity *entity, bool showEntity, bool showBBox,
			Ogre::SceneManager *sceneMgr, const glm::dvec3 &scale, double linMassScale,
			double angMassScale, glm::dvec3 initPos, btCollisionWorld *cW, bool constrained);

	RigidBody(const RigidBody &obj) {
		memcpy(this, &obj, sizeof(obj));
		indices = new unsigned long[index_count];
		memcpy(indices, obj.indices, index_count * sizeof(unsigned long));

		vertices = new glm::dvec3[index_count];
		memcpy(vertices, obj.vertices, vertex_count * sizeof(glm::dvec3));

		collisionObject = new btCollisionObject();
		collisionObject->setUserPointer(this);

		simplifiedConvexShape =
				new btConvexHullShape((const btScalar*)obj.simplifiedConvexShape->getUnscaledPoints(),
				obj.simplifiedConvexShape->getNumPoints());
		collisionObject->setCollisionShape(simplifiedConvexShape);
		collisionObject->getWorldTransform().setOrigin(btVector3(p.x, p.y, p.z));
		collisionObject->getWorldTransform().setRotation(btQuaternion(b2w_rot.x, b2w_rot.y, b2w_rot.z, b2w_rot.w));

		collisionWorld->addCollisionObject(collisionObject);
		std::cout<<"Creating Copy of Rigid Body"<<std::endl;
	}

	~RigidBody() {
		delete []vertices;
		delete []indices;
		delete simplifiedConvexShape;
		collisionWorld->removeCollisionObject(collisionObject);
		delete collisionObject;
		std::cout<<"Destroying Rigid Body"<<std::endl;
	}
};

#endif
