#ifndef __RigidBodySystem_h_
#define __RigidBodySystem_h_

#include "BaseApplication.h"
#include "RigidBody.h"
#include "Contact.h"
#include <btBulletDynamicsCommon.h>

//---------------------------------------------------------------------------

class RigidBodySystem : public BaseApplication
{
public:
	RigidBodySystem() {
		broadphase = 0;
		collisionConfiguration = 0;
		dispatcher = 0;
		collisionWorld = 0;
	};
	~RigidBodySystem() {
		delete collisionWorld;
		delete dispatcher;
		delete collisionConfiguration;
		delete broadphase;
	};
private:
	std::vector<RigidBody> bodies;
	std::vector<Contact> contacts;
	std::map<Ogre::Entity*, unsigned long> pickBody;
	Ogre::ManualObject* lineObject; //Draw the force line when an object is dragged
	glm::dvec3 getSpringForce(glm::dvec3 startPoint, glm::dvec3 endPoint, glm::dvec3 vel) {
		double k = 0.01;
		double c = 0.001;

		glm::dvec3 dx = endPoint - startPoint;
		double l = dx.length();
		glm::normalize(dx);
		double forceMag = k * l - c * glm::dot(dx, vel);

		return dx * forceMag;
	}

	void physicsInit(void);
	btBroadphaseInterface* broadphase;
	btDefaultCollisionConfiguration* collisionConfiguration;
	btCollisionDispatcher* dispatcher;
	btCollisionWorld* collisionWorld;

protected:
    virtual void createScene(void);
    virtual void animate(void);
    virtual void mousePressedRigidBody(OIS::MouseButtonID id);
    virtual void mouseMovedRigidBody(void);
    virtual void mouseReleasedRigidBody(void);
    void addNinja(void);
    void addGround(void);
};

//---------------------------------------------------------------------------

#endif // #ifndef __TutorialApplication_h_

//---------------------------------------------------------------------------
