/*
 * This software is Copyright (c) 2017 Sayantan Datta <std2048 at gmail dot com>
 * and it is hereby released to the general public under the following terms:
 * Redistribution and use in source and binary forms, with or without modification, are permitted for non-profit
 * and non-commericial purposes.
 */

#ifndef __RigidBodySystem_h_
#define __RigidBodySystem_h_

#include "BaseApplication.h"
#include "RigidBody.h"
#include "Contact.h"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <btBulletDynamicsCommon.h>
#include "OgreText.h"

//---------------------------------------------------------------------------
struct ContactInfo {
	float pentrationError;
	unsigned int numContacts;
};

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
	static double dt;
	static double bounce;
	static double mu;
	static double gravity;
private:
	std::vector<RigidBody> bodies;
	std::vector<Contact> contacts;
	std::map<Ogre::Entity*, unsigned long> pickBody;
	Ogre::ManualObject* lineObject; //Draw the force line when an object is dragged
	OgreText *textItem;
	glm::dvec3 getSpringForce(glm::dvec3 startPoint, glm::dvec3 endPoint, glm::dvec3 vel) {
		double k = 0.5;
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
    virtual void keyPressedRigidBody(const OIS::KeyEvent &arg);

private:
    void addNinja(void);
    void addGround(void);
    void addCube();
    void addLight(void);
    void addOverlay(void);

    static bool showBoundingBox;

    static bool captureFrames;

    Ogre::Timer timer;
    unsigned long time;
    void screenCaptureDataProcess();
    std::thread t_screenCapture;
    std::mutex m;
    std::condition_variable cv;
    void screenCaptureDataGenerate();
    std::queue<Ogre::PixelBox> imageBuffer;


    void physicsProcess();
    ContactInfo physicsRun();
    void physicsStart();
    std::thread t_physicsProcess;
    std::mutex m_physics;
    std::mutex m_physics_2;
    std::condition_variable cv_physics;
    std::condition_variable cv_physics_2;
    bool pausePhysics;
    bool physicsSystemLocked;
    bool pauseAnim;

    //Display variables
    float physFPS;
    ContactInfo contactInfo;
    unsigned int nBody;
};

bool RigidBodySystem::captureFrames = false;
bool RigidBodySystem::showBoundingBox = false;

double RigidBodySystem::dt = 0.05;
double RigidBodySystem::bounce = 0.0;
double RigidBodySystem::mu = 0.33;
double RigidBodySystem::gravity = -0.1;

#endif