Videos:
Object Picking: https://youtu.be/e5cPB-T38BQ
Pile of Objects using GPU: https://youtu.be/mK0WW5BIDwI
Pile of Objects using CPU: https://youtu.be/3cQoyWA8VRU
Random pile of objects using GPU: https://youtu.be/Q7vJLqXqCxM
Random pile of objects using CPU:https://youtu.be/j9Sr3QhxbDI

Installing and Running the program:

Install OIS(required for Ogre3d)

Compile Ogrev1.9
Complie Bullet3
Compile glm

This project was built using Eclipse Luna C++ IDE.

Use: g++ -std=c++11

Include paths(-I):
/usr/local/include/OGRE/Overlay
/opt/AMDAPPSDK-2.9-1/include/
/home/sayantan/bullet3-2.86.1/src
/home/sayantan/glm
/usr/local/include/OGRE/
/usr/include/OIS
/home/sayantan/ogre/Samples/Common/include

Library search paths(-L):
/usr/local/lib
/opt/AMDAPPSDK-2.9-1/lib/x86_64
/home/sayantan/bullet3-2.86.1/src/BulletDynamics
/home/sayantan/bullet3-2.86.1/src/BulletCollision
/home/sayantan/bullet3-2.86.1/src/LinearMath
/usr/local/lib/OGRE

Linker flags(-l):
OgreMain
OpenCL
OgreOverlay
OIS
boost_system
BulletDynamics
BulletCollision
LinearMath

Extra fonts:
Extract content of fonts.zip to /home/sayantan/ogre/Samples/Media/fonts/

Add following lines to /home/sayantan/ogre/Samples/Media/materials/scripts/Examples.material

material Examples/Cursor
{
   technique
   {
     pass
     {
       scene_blend alpha_blend
       depth_write off
 
       texture_unit
       {
         colour_op_ex source1 src_manual src_current 1 1 1
         alpha_op_ex source1 src_manual src_current 0.5
       }
     }
   }
}

Also modify the paths inside cfg files if needed.
