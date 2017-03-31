#include "RigidBodySystem.h"
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iomanip>

void RigidBodySystem::addNinja() {
	if (!collisionWorld) {
		std::cout<<"Cannot add Ninja...Init physics first."<<std::endl;
		return;
	}
	Ogre::Entity *entity = mSceneMgr->createEntity("ninja.mesh");
	entity->setCastShadows(true);
	try {
		bodies.push_back(RigidBody(bodies.size(), entity, true, showBoundingBox, mSceneMgr, glm::dvec3(1.5, 1.5, 1.5), 1.0/30.0, 1.0/600, glm::dvec3(0,500,0), collisionWorld, false));
	} catch(std::bad_alloc &xa) {
		std::cerr<<"Couldn't Reallocate RigidBody stack"<<std::endl;
		exit(0);
	}
	pickBody[entity] = bodies.size() - 1;
}

void RigidBodySystem::addCube() {
	if (!collisionWorld) {
		std::cout<<"Cannot add Cube...Init physics first."<<std::endl;
		return;
	}

	Ogre::Entity* entity = mSceneMgr->createEntity("ColourCube");
	entity->setMaterialName("Test/ColourTest");
	entity->setCastShadows(true);
	try {
		bodies.push_back(RigidBody(bodies.size(), entity, true, showBoundingBox, mSceneMgr, glm::dvec3(30.0), 20.0, 1.0, glm::dvec3(0, 500, 0), collisionWorld, false));
	} catch(std::bad_alloc &xa) {
		std::cerr<<"Couldn't Reallocate RigidBody stack"<<std::endl;
		exit(0);
	}
	pickBody[entity] = bodies.size() - 1;
}

void RigidBodySystem::addGround() {
	if (!collisionWorld) {
		std::cout<<"Cannot add Ground...Init physics first."<<std::endl;
		return;
	}

	Ogre::Plane plane(Ogre::Vector3::UNIT_Y, 0);
	Ogre::MeshManager::getSingleton().createPlane(
		  "ground",
	Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME,
		  plane,
		  2000, 2000, 1, 1,
		  true,
		  1, 5, 5,
	Ogre::Vector3::UNIT_Z);
	Ogre::Entity* animEntity = mSceneMgr->createEntity("ground");
	animEntity->setMaterialName("Examples/Rockwall");

	mSceneMgr->getRootSceneNode()->createChildSceneNode(Ogre::Vector3(0, 1.0, 0))->attachObject(animEntity);

	/* For physics use a simpler 3D cuboid as ground*/
	Ogre::Entity* phyEntity = mSceneMgr->createEntity("ColourCube");
	phyEntity->setMaterialName("Test/ColourTest");
	try {
		bodies.push_back(RigidBody(bodies.size(), phyEntity, false, showBoundingBox, mSceneMgr, glm::dvec3(2000.0, 1.0, 2000.0), 1.0/300.0, 1.0/300.0, glm::dvec3(0), collisionWorld, true));
	} catch(std::bad_alloc &xa) {
		std::cerr<<"Couldn't Reallocate RigidBody stack"<<std::endl;
		exit(0);
	}
	pickBody[phyEntity] = bodies.size() - 1;
}

void RigidBodySystem::physicsInit(){
	broadphase = new btDbvtBroadphase();
	collisionConfiguration = new btDefaultCollisionConfiguration();
	dispatcher = new btCollisionDispatcher(collisionConfiguration);
	collisionWorld = new btCollisionWorld(dispatcher, broadphase, collisionConfiguration);

	contacts.reserve(50);
}

void RigidBodySystem::addLight(void) {
	mSceneMgr->setAmbientLight(Ogre::ColourValue(0.01, 0.01, 0.01));
	Ogre::Light* directionalLight = mSceneMgr->createLight("DirectionalLight");
	directionalLight->setType(Ogre::Light::LT_DIRECTIONAL);
	directionalLight->setDiffuseColour(Ogre::ColourValue(1, 1, 1));
	directionalLight->setSpecularColour(Ogre::ColourValue(1, 1, 1));
	directionalLight->setDirection(Ogre::Vector3(0, -1, 1));

	mSceneMgr->setShadowTechnique(Ogre::SHADOWTYPE_STENCIL_ADDITIVE);
}

void RigidBodySystem::addOverlay() {
	Ogre::OverlayManager& overlayManager = Ogre::OverlayManager::getSingleton();
	// Create an overlay
	Ogre::Overlay* crosshair = overlayManager.create("crosshair");
	// Create a panel
	Ogre::OverlayContainer* panel = static_cast<Ogre::OverlayContainer*>( overlayManager.createOverlayElement( "Panel", "PanelName" ) );
	panel->setPosition( 0.495, 0.495);
	panel->setDimensions(0.01, 0.01);
	panel->setMaterialName("Examples/Cursor");
	// Add the panel to the overlay
	crosshair->add2D( panel );
	// Show the overlay
	crosshair->show();
}

void RigidBodySystem::createScene(void)
{
	physicsInit();
	bodies.reserve(10);
	// Create your scene here :)
	addCube();
	addGround();

	addLight();

	addOverlay();

    lineObject =  mSceneMgr->createManualObject("line");
    // NOTE: The second parameter to the create method is the resource group the material will be added to.
    // If the group you name does not exist (in your resources.cfg file) the library will assert() and your program will crash
    Ogre::MaterialPtr lineMaterial = Ogre::MaterialManager::getSingleton().create("lineMaterial","General");
    lineMaterial->setReceiveShadows(false);
    lineMaterial->getTechnique(0)->setLightingEnabled(true);
    lineMaterial->getTechnique(0)->getPass(0)->setDiffuse(1,0,0,0);
    lineMaterial->getTechnique(0)->getPass(0)->setAmbient(1,0,0);
    lineMaterial->getTechnique(0)->getPass(0)->setSelfIllumination(1,0,0);
    lineMaterial->getTechnique(0)->getPass(0)->setPointMaxSize(6);
    //myManualObjectMaterial->dispose();  // dispose pointer, not the material
    Ogre::SceneNode* lineNode = mSceneMgr->getRootSceneNode()->createChildSceneNode();
    lineObject->begin("lineMaterial", Ogre::RenderOperation::OT_LINE_LIST);
   	lineObject->position(0, 0, 0);
  	lineObject->position(0, 0, 0);
   	lineObject->end();
    lineNode->attachObject(lineObject);

    timer = Ogre::Timer();
    time = timer.getMilliseconds();

    if (captureFrames)
    	t_screenCapture = std::thread(&RigidBodySystem::screenCaptureDataProcess, this);
}

void RigidBodySystem::screenCaptureDataGenerate() {
	 int left, top, width, height;
	 Ogre::Viewport *vp = mWindow->getViewport(0);
	 vp->getActualDimensions(left, top, width, height);

	Ogre::PixelFormat format = Ogre::PF_BYTE_RGBA;
	unsigned int bytesPerPixel = Ogre::PixelUtil::getNumElemBytes(format);

	unsigned char *data = new unsigned char [width * height * bytesPerPixel];
	Ogre::Box extents(left, top, left + width, top + height);
	Ogre::PixelBox pb(extents, format, data);

	//printf("PixelBox: %d, %d, w: %d, h: %d\n", pb.left, pb.right, pb.getWidth(), pb.getHeight());

	mWindow->copyContentsToMemory(pb);

	{
		//Hold mutex
		std::lock_guard<std::mutex> lk(m);
		//Process critical data
		imageBuffer.push(pb);
	} // Mutex released when lk is destroyed

	// Notify the other thread that the critical data is modified and recheck the condition if it is waiting.
	cv.notify_one();

}

void RigidBodySystem::animate() {
	double dt = 0.05;
	double bounce = 0.0;
	double mu = 0.33;

	if (mouseButtonDown) {
		unsigned long i = pickBody[selectedEntity];

		glm::dvec4 startWorld = bodies[i].getBodyToWorld(glm::dvec4(startPoint.x, startPoint.y, startPoint.z, 1));
		lineObject->beginUpdate(0);
		lineObject->position(startWorld.x, startWorld.y, startWorld.z);
		lineObject->position(endPoint);
		lineObject->end();

		bodies[i].applyForce(glm::dvec3(startWorld),
				getSpringForce(glm::dvec3(startWorld), glm::dvec3(endPoint.x, endPoint.y, endPoint.z),
				bodies[i].getContactVelocity(glm::dvec3(startWorld))));
	}
	for (size_t i = 0; i < bodies.size(); i++)
		bodies[i].applyForce(glm::dvec3(0, -10, 0));

	collisionWorld->performDiscreteCollisionDetection();

	int numManifolds = collisionWorld->getDispatcher()->getNumManifolds();

	unsigned int numContacts = 1;
	for (int i = 0; i < numManifolds; i++)
		numContacts *= collisionWorld->getDispatcher()->getManifoldByIndexInternal(i)->getNumContacts();

	if (numContacts > contacts.size()) {
		try {
			contacts.reserve(numContacts * 2);
		} catch(std::bad_alloc &xa) {
			std::cerr<<"Couldn't Reallocate Contact stack"<<std::endl;
			exit(0);
		}
	}

	numContacts = 0;
	for (int i = 0; i < numManifolds; i++) {
		btPersistentManifold* contactManifold = collisionWorld->getDispatcher()->getManifoldByIndexInternal(i);
		const btCollisionObject* obA = static_cast<const btCollisionObject*>(contactManifold->getBody0());
		const btCollisionObject* obB = static_cast<const btCollisionObject*>(contactManifold->getBody1());
		contactManifold->refreshContactPoints(obA->getWorldTransform(), obB->getWorldTransform());
		int _numContacts = contactManifold->getNumContacts();
		//For each contact point in that manifold
	    for (int j = 0; j < _numContacts; j++) {
	      //Get the contact information
	        btManifoldPoint& pt = contactManifold->getContactPoint(j);
	        btVector3 contactPoint = (pt.getPositionWorldOnB());
	        contacts[numContacts] = Contact((RigidBody *)obA->getUserPointer(), (RigidBody *)obB->getUserPointer(),
	        		glm::dvec3(contactPoint.getX(), contactPoint.getY(), contactPoint.getZ()),
					glm::dvec3(-pt.m_normalWorldOnB.getX(), -pt.m_normalWorldOnB.getY(), -pt.m_normalWorldOnB.getZ()), bounce, dt);
	        numContacts++;
	    }
	}

	unsigned int contactPow2 = numContacts;
	contactPow2--;
	contactPow2 |= contactPow2 >> 1;
	contactPow2 |= contactPow2 >> 2;
	contactPow2 |= contactPow2 >> 4;
	contactPow2 |= contactPow2 >> 8;
	contactPow2 |= contactPow2 >> 16;

	srand(std::time(NULL));
	for (int j = 0; j < 100 && numContacts; j++) {

		for (unsigned int i = 0; i < numContacts; i++)
			contacts[i].processed = false;

		for (unsigned int i = 0; i < (numContacts>>1); i++) {
			unsigned int randNum = std::rand() & contactPow2;
			if (randNum >= numContacts) randNum >>= 2;
			if (!contacts[randNum].processed)
				contacts[randNum].processContact(mu);
		}

		for (unsigned int i = 0; i < numContacts; i++) {
			if (!contacts[i].processed)
				contacts[i].processContact(mu);
		}

		/*bool check = true;
		for (unsigned int i = 0; i < numContacts; i++)
			check &= contacts[i].processed;

		if (!check)
			std::cout<<"BACHAO"<<std::endl;*/

	}

	for (size_t i = 0; i < bodies.size() && numContacts; i++)
			bodies[i].updateVelocity();


	for (size_t i = 0; i < bodies.size(); i++)
		bodies[i].advanceTime(dt);

	if (captureFrames && (timer.getMilliseconds() - time) > 33) {
		time = timer.getMilliseconds();
		screenCaptureDataGenerate();
	}
}

inline std::string pad(int n, int len) {
    std::string result(len--, '0');
    for (int val=(n<0)?-n:n; len>=0&&val!=0; --len,val/=10)
       result[len]='0'+val%10;
    if (len>=0&&n<0) result[0]='-';
    return result;
}

// Runs on a separate thread
void RigidBodySystem::screenCaptureDataProcess() {
	static long int sequence = 0;
	for (;;) {
		// Hold mutex
		std::unique_lock<std::mutex> lk(m);

		/*
		 * wait if condition is not satisfied and release mutex while waiting.
		 * Recheck the condition if notified by main thread.
		 */
		cv.wait(lk,  [this](){return imageBuffer.size() > 0;});
		// mutex is re acquired when wait returns.

		// Process critical data
		Ogre::PixelBox pb = imageBuffer.front();
		imageBuffer.pop();
		lk.unlock(); // Release mutex

		Ogre::Image finalImage;
		finalImage = finalImage.loadDynamicImage(static_cast<unsigned char*>(pb.data), pb.getWidth(), pb.getHeight(), pb.format);
		std::string s = "Stills/img" + pad(sequence++, 10) + ".jpg"; // jpg small size and low processing time.
		finalImage.save(s);

		delete []static_cast<unsigned char*>(pb.data);
	}
}

void RigidBodySystem::mousePressedRigidBody(OIS::MouseButtonID id) {
	if (id == OIS::MB_Right) {
		addCube();
	}
	else {
		//Object Picking
		Ogre::Ray ray = this->mCamera->getCameraToViewportRay((float) 0.5, (float) 0.5);

		raySceneQuery->setRay(ray);
		raySceneQuery->setSortByDistance(true);
		raySceneQuery->setQueryTypeMask(Ogre::SceneManager::ENTITY_TYPE_MASK);

		Ogre::RaySceneQueryResult& result = raySceneQuery->execute();
		Ogre::RaySceneQueryResult::iterator it = result.begin();

		for ( ; it != result.end(); it++) {
			bool mMovableFound =
			it->movable &&
			it->movable->getName() != "line" &&
			it->movable->getName() != "PlayerCam";

			if (mMovableFound) {
				selectedEntity = (Ogre::Entity *)it->movable;
				distance = it->distance;
				startPoint = ray.getPoint(distance);
				endPoint = startPoint;
				mouseButtonDown = true;
				glm::dvec4 body = bodies[pickBody[selectedEntity]].getWorldToBody(glm::dvec4(startPoint.x, startPoint.y, startPoint.z, 1));
				startPoint = Ogre::Vector3(body.x, body.y, body.z);
				std::cout<<"Object Picked:"<<it->movable->getName()<<" "<<pickBody[selectedEntity]<<std::endl;
				break;
			}
		}
	}
}

void RigidBodySystem::mouseMovedRigidBody() {
	if (mouseButtonDown) {
		Ogre::Ray ray = this->mCamera->getCameraToViewportRay((float) 0.5, (float) 0.5);
	    endPoint = ray.getPoint(distance);
	}
}

void RigidBodySystem::mouseReleasedRigidBody() {
	if (mouseButtonDown) {
		selectedEntity = NULL;
		distance = 0;
		startPoint = Ogre::Vector3(0,0,0);
		endPoint = Ogre::Vector3(0,0,0);
		mouseButtonDown = false;
		lineObject->beginUpdate(0);
		lineObject->position(0, 0, 0);
		lineObject->position(0, 0, 0);
		lineObject->end();
		std::cout<<"Object Released"<<std::endl;
	}
}

//---------------------------------------------------------------------------

#if OGRE_PLATFORM == OGRE_PLATFORM_WIN32
#define WIN32_LEAN_AND_MEAN
#include "windows.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if OGRE_PLATFORM == OGRE_PLATFORM_WIN32
    INT WINAPI WinMain(HINSTANCE hInst, HINSTANCE, LPSTR strCmdLine, INT)
#else
    int main(int argc, char *argv[])
#endif
    {
        // Create application object
        RigidBodySystem app;

        try {
            app.go();
        } catch(Ogre::Exception& e)  {
#if OGRE_PLATFORM == OGRE_PLATFORM_WIN32
            MessageBox(NULL, e.getFullDescription().c_str(), "An exception has occurred!", MB_OK | MB_ICONERROR | MB_TASKMODAL);
#else
            std::cerr << "An exception has occurred: " <<
                e.getFullDescription().c_str() << std::endl;
#endif
        }

        return 0;
    }

#ifdef __cplusplus
}
#endif

//---------------------------------------------------------------------------
