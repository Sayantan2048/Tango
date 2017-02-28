#include "RigidBody.h"

static void getMeshInformation(Ogre::Mesh* mesh,
                        size_t &vertex_count,
                        Ogre::Vector3* &vertices,
                        size_t &index_count,
                        unsigned long* &indices,
                        const Ogre::Vector3 &position,
                        const Ogre::Quaternion &orient,
                        const Ogre::Vector3 &scale);

RigidBody::RigidBody(unsigned long int index, Ogre::Entity *entity, Ogre::SceneManager *sceneMgr, double massScale, glm::dvec3 initPos, btCollisionWorld *cW, bool constrained) {
	Ogre::Vector3* vertices;

	getMeshInformation((entity->getMesh()).get(), vertex_count, vertices, index_count, indices, Ogre::Vector3(0, 0, 0), Ogre::Quaternion::IDENTITY, Ogre::Vector3(1,1,1));

	glm::dvec3 cm(0.0, 0.0, 0.0);
	double mass = 0;

	for (size_t i = 0; i < index_count; i = i + 3) {
		unsigned long t1 = indices[i];
		unsigned long t2 = indices[i + 1];
		unsigned long t3 = indices[i + 2];

		Ogre::Vector3 v1 = vertices[t1];
		Ogre::Vector3 v2 = vertices[t2];
		Ogre::Vector3 v3 = vertices[t3];

		Ogre::Vector3 centroid((v1+v2+v3)/3.0);
		double area = 0.5 * ((v1 - v2).crossProduct(v1 - v3)).length() * massScale;
		area = area > 0 ? area: -area;
		mass += area;
		cm += glm::dvec3(area * centroid.x, area * centroid.y, area * centroid.z);
	}

	iMass = 1/mass;
	cm *= iMass;

	if (constrained)
		iMass = 0;

	this->vertices = new glm::dvec3[vertex_count];

	for (size_t i = 0; i < vertex_count; i++) {
		this->vertices[i] = glm::dvec3(vertices[i].x - cm.x, vertices[i].y - cm.y, vertices[i].z - cm.z);
	}

	double Ixx = 0;
	double Iyy = 0;
	double Izz = 0;
	double Ixy = 0;
	double Ixz = 0;
	double Iyz = 0;
	for (size_t i = 0; i < index_count && !constrained; i = i + 3) {
		unsigned long t1 = indices[i];
		unsigned long t2 = indices[i + 1];
		unsigned long t3 = indices[i + 2];

		glm::dvec3 v1 = this->vertices[t1];
		glm::dvec3 v2 = this->vertices[t2];
		glm::dvec3 v3 = this->vertices[t3];

		glm::dvec3 centroid((v1+v2+v3)/3.0);

		double area = 0.5 * glm::cross((v1 - v2),(v1 - v3)).length() * massScale;
		area = area > 0 ? area: -area;

		Ixx += area * (centroid.z * centroid.z + centroid.y * centroid.y);
		Iyy += area * (centroid.x * centroid.x + centroid.z * centroid.z);
		Izz += area * (centroid.x * centroid.x + centroid.y * centroid.y);

		Ixy -= area * centroid.x * centroid.y;
		Ixz -= area * centroid.x * centroid.z;
		Iyz -= area * centroid.z * centroid.y;
	}

	iT_0 = glm::dmat3x3(Ixx, Ixy, Ixz, Ixy, Iyy, Iyz, Ixz, Iyz, Izz);
	iiT_0 = constrained?iT_0:glm::inverse(iT_0);

	p = cm + initPos;
	v = glm::dvec3(0,0,0);
	w = glm::dvec3(0.00,0,0);
	b2w_rot = glm::dquat(1, 0, 0, 0);
	f = glm::dvec3(0,0,0);
	t = glm::dvec3(0,0,0);

	deltaV = glm::dvec3(0,0,0);
	deltaW = glm::dvec3(0,0,0);

	delete[] vertices;

	node = sceneMgr->getRootSceneNode()->createChildSceneNode();
	Ogre::SceneNode *n = node->createChildSceneNode(Ogre::Vector3(-cm.x, -cm.y, -cm.z));
	n->attachObject(entity);
	n->showBoundingBox(true);

	this->constrained = constrained;
	this->index = index;

	std::cout<<"Vertices in mesh:"<< vertex_count<<std::endl;
	std::cout<<"Triangles in mesh:"<< index_count / 3<<std::endl;
	std::cout<<"CM"<<cm.x<<" "<<cm.y<<" " <<cm.z<<std::endl;
	std::cout<<"Mass"<<mass<<std::endl;

	collisionObject = new btCollisionObject();
	collisionObject->setUserPointer(this);

	btConvexHullShape complexHull = btConvexHullShape();
	for (size_t i = 0; i < vertex_count; i++) {
		//Vertices are copied by the function
		complexHull.addPoint(btVector3(this->vertices[i].x, this->vertices[i].y, this->vertices[i].z));
	}

	//create a hull approximation
	//btShapeHull hull = btShapeHull(&complexHull);
	//btScalar margin = complexHull.getMargin();
	//hull.buildHull(margin);
	//simplifiedConvexShape = new btConvexHullShape((const btScalar*)hull.getVertexPointer(), hull.numVertices());
	simplifiedConvexShape = new btConvexHullShape(complexHull);
	collisionObject->setCollisionShape(simplifiedConvexShape);

	cW->addCollisionObject(collisionObject);
	collisionWorld = cW;

	updateTransform();
}

void RigidBody::updateTransform() {
	/* Call after initializing collisionObject */
	b2w = glm::toMat4(b2w_rot);
	glm::dmat3x3 rot = glm::dmat3x3(b2w);
	iT = rot * iT_0 * glm::transpose(rot);
	iiT = rot * iiT_0 * glm::transpose(rot);
	b2w[3] = glm::dvec4(p.x, p.y, p.z, 1.0);
	w2b = glm::inverse(b2w);

	collisionObject->getWorldTransform().setOrigin(btVector3(p.x, p.y, p.z));
	collisionObject->getWorldTransform().setRotation(btQuaternion(b2w_rot.x, b2w_rot.y, b2w_rot.z, b2w_rot.w));
}

void RigidBody::advanceTime(double dt) {
	v += f * iMass * dt;
	w += dt * iiT * t;

	p += dt * v;
	b2w_rot += dt * glm::dquat(0.5 * glm::dot(glm::dvec3(-b2w_rot.x, -b2w_rot.y, -b2w_rot.z) , w),
				0.5 * glm::dot(glm::dvec3(b2w_rot.w, b2w_rot.z, -b2w_rot.y) , w),
				0.5 * glm::dot(glm::dvec3(-b2w_rot.z, b2w_rot.w, b2w_rot.x) , w),
				0.5 * glm::dot(glm::dvec3(b2w_rot.y, -b2w_rot.x, b2w_rot.w) , w));

	glm::normalize(b2w_rot);
	updateTransform();
	node->setPosition(p.x, p.y, p.z);
	node->setOrientation(b2w_rot.w, b2w_rot.x,b2w_rot.y, b2w_rot.z);

	f = glm::dvec3(0,0,0);
	t = glm::dvec3(0,0,0);
}

void RigidBody::applyForce(glm::dvec3 contactPoint, glm::dvec3 force) {
	glm::dvec3 r = contactPoint - p;
	t = glm::cross(r, force);
	f = force;

}

glm::dvec4 RigidBody::getWorldToBody(glm::dvec4 world) {
	return w2b * world;
}

glm::dvec4 RigidBody::getBodyToWorld(glm::dvec4 body) {
	return b2w * body;
}

glm::dvec3 RigidBody::getContactVelocity(glm::dvec3 contactPoint) {
	glm::dvec3 r = contactPoint - p;
	return v + glm::cross(w, r);
}

static void getMeshInformation(Ogre::Mesh* mesh,
                        size_t &vertex_count,
                        Ogre::Vector3* &vertices,
                        size_t &index_count,
                        unsigned long* &indices,
                        const Ogre::Vector3 &position,
                        const Ogre::Quaternion &orient,
                        const Ogre::Vector3 &scale)
{
    bool added_shared = false;
    size_t current_offset = 0;
    size_t shared_offset = 0;
    size_t next_offset = 0;
    size_t index_offset = 0;

    vertex_count = index_count = 0;

    // Calculate how many vertices and indices we're going to need
    for ( unsigned short i = 0; i < mesh->getNumSubMeshes(); ++i)
    {
        Ogre::SubMesh* submesh = mesh->getSubMesh(i);
        // We only need to add the shared vertices once
        if(submesh->useSharedVertices)
        {
            if( !added_shared )
            {
                vertex_count += mesh->sharedVertexData->vertexCount;
                added_shared = true;
            }
        }
        else
        {
            vertex_count += submesh->vertexData->vertexCount;
        }
        // Add the indices
        index_count += submesh->indexData->indexCount;
    }

    // Allocate space for the vertices and indices
    vertices = new Ogre::Vector3[vertex_count];
    indices = new unsigned long[index_count];

    added_shared = false;

    // Run through the submeshes again, adding the data into the arrays
    for (unsigned short i = 0; i < mesh->getNumSubMeshes(); ++i)
    {
        Ogre::SubMesh* submesh = mesh->getSubMesh(i);

        Ogre::VertexData* vertex_data = submesh->useSharedVertices ? mesh->sharedVertexData : submesh->vertexData;

        if ((!submesh->useSharedVertices) || (submesh->useSharedVertices && !added_shared))
        {
            if(submesh->useSharedVertices)
            {
                added_shared = true;
                shared_offset = current_offset;
            }

            const Ogre::VertexElement* posElem =
                vertex_data->vertexDeclaration->findElementBySemantic(Ogre::VES_POSITION);

            Ogre::HardwareVertexBufferSharedPtr vbuf =
                vertex_data->vertexBufferBinding->getBuffer(posElem->getSource());

            unsigned char* vertex =
                static_cast<unsigned char*>(vbuf->lock(Ogre::HardwareBuffer::HBL_READ_ONLY));

            // There is _no_ baseVertexPointerToElement() which takes an Ogre::Real or a double
            //  as second argument. So make it float, to avoid trouble when Ogre::Real will
            //  be comiled/typedefed as double:
            //Ogre::Real* pReal;
            float* pReal;

            for( size_t j = 0; j < vertex_data->vertexCount; ++j, vertex += vbuf->getVertexSize())
            {
                posElem->baseVertexPointerToElement(vertex, &pReal);
                Ogre::Vector3 pt(pReal[0], pReal[1], pReal[2]);
                vertices[current_offset + j] = (orient * (pt * scale)) + position;
            }

            vbuf->unlock();
            next_offset += vertex_data->vertexCount;
        }

        Ogre::IndexData* index_data = submesh->indexData;
        size_t numTris = index_data->indexCount / 3;
        Ogre::HardwareIndexBufferSharedPtr ibuf = index_data->indexBuffer;

        bool use32bitindexes = (ibuf->getType() == Ogre::HardwareIndexBuffer::IT_32BIT);

        unsigned long* pLong = static_cast<unsigned long*>(ibuf->lock(Ogre::HardwareBuffer::HBL_READ_ONLY));
        unsigned short* pShort = reinterpret_cast<unsigned short*>(pLong);

        size_t offset = (submesh->useSharedVertices)? shared_offset : current_offset;

        if ( use32bitindexes )
        {
            for ( size_t k = 0; k < numTris*3; ++k)
            {
                indices[index_offset++] = pLong[k] + static_cast<unsigned long>(offset);
            }
        }
        else
        {
            for ( size_t k = 0; k < numTris*3; ++k)
            {
                indices[index_offset++] = static_cast<unsigned long>(pShort[k]) +
                                          static_cast<unsigned long>(offset);
            }
        }

        ibuf->unlock();
        current_offset = next_offset;
    }
}
