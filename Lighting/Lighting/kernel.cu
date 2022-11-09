#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_gl_interop.h"


#include <thread>
#include <chrono>
#include <stdio.h>
#include <iostream>
#include "Light.h"
#include <vector>

const int LIGHT_SAMPLE_POINTS = 10;
const float LIGHT_SAMPLE_POINT_RADIUS = 15.0;
const int LIGHT_COUNT = 2;
const int OBSTACLE_COUNT = 2;

__device__
typedef struct light
{
    float lightx;
    float lighty;
    float startingIntensity;
    float radius;
    //float visibilityLeftRotations[10];
    //float visibilityRightRotations[10];
    //float visibilityRadius[10];

    //float testxpoints[LIGHT_SAMPLE_POINTS];
   // float testypoints[LIGHT_SAMPLE_POINTS];
};

__device__
typedef struct obstacle
{
    int x;
    int y;
    int width;
    int height;
};

//float* getMapData();
void errorCheck(int errorVal);
void endProgram();
void processKey(GLFWwindow* window);

//const int MAP_WIDTH = 640;
//const int MAP_HEIGHT = 480;

const int MAP_WIDTH = 1500;
const int MAP_HEIGHT = 700;

GLFWwindow* window;

GLuint lightMapTexture;
cudaGraphicsResource* lightMapTextureResource;
uchar4* g_dstBuffer = NULL;
size_t g_BufferSize = 0;

light* h_Lights;
int lightIndex = 0;
light* d_Lights;
obstacle* h_obstacles;
int obstacleIndex = 0;
obstacle* d_obstacles;

__device__
float amax(float a, float b) {
    if (a >= b) {
        return a;
    }

    return b;
}

__device__
float amin(float a, float b) {
    if (a <= b) {
        return a;
    }

    return b;
}


__device__
float distance(float x1, float y1, float x2, float y2) {
    float xdif = x1 - x2;
    float ydif = y1 - y2;
    //return std::sqrt((xdif * xdif) + (ydif * ydif));
    return sqrt((xdif * xdif) + (ydif * ydif));
}

__device__
float rotation(float x1, float y1, float x2, float y2) {
    float xdif, ydif;

    xdif = x2 - x1;
    ydif = y2 - y1;
    

    float rot = atan(abs(ydif) / abs(xdif));

    if (xdif >= 0.0) {
        if (ydif >= 0.0) {
            return rot;
        }
        else
        {
            return (2 * 3.14) - rot;
        }
    }
    else
    {
        if (ydif >= 0.0) {
            return 3.14 - rot;
        }
        else
        {
            return 3.14 + rot;
        }
    }

}

__device__
bool pointOnLine(float l1x, float l1y, float l2x, float l2y, float px, float py)
{
    float leftx, rightx,lefty,righty;
    leftx = amin(l1x, l2x);
    rightx = amax(l1x, l2x);

    if (!((px >= leftx) && (px <= rightx))) {
        return false;
    }

    if (l1x <= l2x) {
        lefty = l1y;
        righty = l2y;
    }
    else
    {
        lefty = l2y;
        righty = l1y;
    }

    float pct = (px - leftx) / (rightx - leftx);
    float targetY = ((1 - pct) * lefty) + (pct * righty);

    return abs(py - targetY) < 4.0;
}

__device__
inline bool withinBox(float bx, float by, float bw, float bh, float tx, float ty) {
    return ((tx > bx) && (tx < (bx + bw)) && (ty > by) && (ty < (by + bh)));
}

__device__
bool clearPath(int targetx, int targety, light tlight, obstacle* obstacleList, int obstacleCount)
{
    float SAMPLE_POINT_RADIUS = 15.0;
    float leftx, rightx, topy, bottomy;
    leftx = 99999.0;
    rightx = -999999.0;
    topy = 9999999;
    bottomy = -99999.0;
    leftx = amin(leftx, tlight.lightx - SAMPLE_POINT_RADIUS);
    leftx = amin(leftx, targetx);
    rightx = amax(rightx, tlight.lightx + SAMPLE_POINT_RADIUS);
    rightx = amax(rightx, targetx);
    topy = amin(topy, tlight.lighty - SAMPLE_POINT_RADIUS);
    topy = amin(topy, targety);
    bottomy = amax(bottomy, tlight.lighty + SAMPLE_POINT_RADIUS);
    bottomy = amax(bottomy, targety);

    for (int obstacleIndex = 0; obstacleIndex < obstacleCount; obstacleIndex++) {
        if (withinBox(leftx, topy, rightx - leftx, bottomy - topy, obstacleList[obstacleIndex].x, obstacleList[obstacleIndex].y)) {
            return false;
        }

        if (withinBox(leftx, topy, rightx - leftx, bottomy - topy, obstacleList[obstacleIndex].x + obstacleList[obstacleIndex].width, obstacleList[obstacleIndex].y)) {
            return false;
        }

        if (withinBox(leftx, topy, rightx - leftx, bottomy - topy, obstacleList[obstacleIndex].x + obstacleList[obstacleIndex].width, obstacleList[obstacleIndex].y + obstacleList[obstacleIndex].height)) {
            return false;
        }

        if (withinBox(leftx, topy, rightx - leftx, bottomy - topy, obstacleList[obstacleIndex].x, obstacleList[obstacleIndex].y + obstacleList[obstacleIndex].height)) {
            return false;
        }
    }

    return true;
}

__device__
float getLightValue(float startingIntensity, int lightx, int lighty,float lightRadius,int targetx, int targety) {
    float xdif, ydif;
    xdif = lightx - targetx;
    ydif = lighty - targety;
    float distance = sqrt((xdif * xdif) + (ydif * ydif));
    float pct = distance / lightRadius;
    pct *= 4.0;
    float multiplier = 1.0 / (1.0 + pct + (pct * pct));
    multiplier -= 0.048;
    
    if (distance > lightRadius) {
        return 0.0;
    }

    if (multiplier < 0.0) {
        return 0.0;
    }

    return startingIntensity * multiplier;
    //return startingIntensity - (startingIntensity * (distance / lightRadius));
}

__device__
bool rayTrace(float x1, float y1, float x2, float y2, obstacle* obstacleList, int obstacleCount) {
    float pointsPerDistance = 2.0;
    int points = (int) (distance(x1, y1, x2, y2) * pointsPerDistance);
    float xstep, ystep;
    float cx, cy;

    xstep = (x2 - x1) / ((float)points);
    ystep = (y2 - y1) / ((float)points);

    cx = x1;
    cy = y1;

    for (int i = 0; i < points; i++) {
        for (int obstacleIndex = 0; obstacleIndex < obstacleCount; obstacleIndex++) {
            if (withinBox(obstacleList[obstacleIndex].x, obstacleList[obstacleIndex].y, obstacleList[obstacleIndex].width, obstacleList[obstacleIndex].height, cx, cy)) {
                return false;
            }

            cx += xstep;
            cy += ystep;
        }
    }

    return true;
}

__device__
float fullLight(int targetx, int targety, light tlight, obstacle* obstacleList, int obstacleCount) {
    float tx = (float) targetx;
    float ty = (float) targety;

    float pointToLightRotation = rotation( tlight.lightx, tlight.lighty,tx,ty);
    float firstRotation, secondRotation;

    firstRotation = pointToLightRotation + (3.14 * 0.5);
    
    if (firstRotation > (2 * 3.14)) {
        firstRotation -= (2 * 3.14);
    }

    secondRotation = firstRotation + 3.14;

    if (secondRotation > (2 * 3.14)) {
        secondRotation -= 2 * 3.14;
    }

    float xpoints[2];
    float ypoints[2];

    xpoints[0] = tlight.lightx + (15.0 * cos(firstRotation));
    ypoints[0] = tlight.lighty + (15.0 * sin(firstRotation));

    xpoints[1] = tx + (15.0 * cos(secondRotation));
    ypoints[1] = ty + (15.0 * sin(secondRotation));

    if (!(rayTrace(xpoints[0], ypoints[0], tx, ty, obstacleList, obstacleCount))) {
        return false;
    }

    if (!(rayTrace(xpoints[1], ypoints[1], tx, ty, obstacleList, obstacleCount))) {
        return false;
    }

    return true;
}

__device__
float getLightAmount(int targetx, int targety, light tlight,obstacle* obstacleList, int obstacleCount) {
    int lightSamplePoints = 10;

    light thelight = tlight;
    
    const float SAMPLE_POINTS_PER_DISTANCE = 0.7;
    
    float originalIntensity = getLightValue(thelight.startingIntensity, thelight.lightx, thelight.lighty, thelight.radius, targetx, targety);
    float intensityPerConnection = originalIntensity / (float)lightSamplePoints;
    
    if ((intensityPerConnection) < 0.00001) {
        return (float) 0.0;
    }

    if (clearPath(targetx, targety, tlight, obstacleList, obstacleCount)) {
        return originalIntensity;
    }

    //if (fullLight(targetx, targety, tlight, obstacleList, obstacleCount)) {
    //    return originalIntensity;
    //}

    


    float currentx, currenty,xinc,yinc;
    currentx = currenty = xinc = yinc = 0.0;
    //return 0.5;
    int samplePoints = (int) (distance(targetx, targety, thelight.lightx, thelight.lighty) * SAMPLE_POINTS_PER_DISTANCE);
    float result = 0.0;

    bool hitObstacle = false;

    //return 0.5;

    for (int i = 0; i < lightSamplePoints; i++) {
        currentx = targetx;
        currenty = targety;

        float rotation = ((float) i / (float) lightSamplePoints) * 2.0 * 3.14;
        
        float testXPoint = thelight.lightx + 15.0 * cos(rotation);
        float testYPoint = thelight.lighty + 15.0 * sin(rotation);

        xinc = (testXPoint - targetx) / (float) samplePoints;
        yinc = (testYPoint - targety) / (float) samplePoints;
        float intensityAtTestPoint = getLightValue(thelight.startingIntensity, thelight.lightx, thelight.lighty, thelight.radius, testXPoint, testYPoint);
        //float finalIntensity = getLightValue(intensityAtTestPoint, thelight.testxpoints[i], thelight.testypoints[i], thelight.radius, targetx, targety);
        //finalIntensity /= (float) lightSamplePoints;

        hitObstacle = false;

        for (int testPoint = 0;testPoint < samplePoints;testPoint++) {
            for (int obstacleIndex = 0; obstacleIndex < obstacleCount; obstacleIndex++) {
                if (withinBox(obstacleList[obstacleIndex].x, obstacleList[obstacleIndex].y, obstacleList[obstacleIndex].width, obstacleList[obstacleIndex].height, currentx, currenty))
                {
                    hitObstacle = true;
                    break;
                }
            }

            if (hitObstacle) {
                break;
            }

            currentx += xinc;
            currenty += yinc;
        }

        if (!(hitObstacle)) {
            result += intensityPerConnection;
            //result += finalIntensity;
        }
    }

    return result;
}

__global__ void calculations(float* mapdata,light* firstLight,obstacle* obstacles)
{   
    int lightCount = 1;// (*lcount);
    int obstacleCount = 1;//(*oCount);
    //int i = threadIdx.x;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    int y = (int)(i / 640);
    int x = i - (y * 640);

    if (i < 0 || i >(640 * 480)) {
        return;
    }

    float startingIntensity = 0.0;
    //printf("%d", sizeof(firstLight));
    //printf("%d", sizeof(*firstLight));
    
    for (int lightIndex = 0; lightIndex < lightCount; lightIndex++) {
        startingIntensity = firstLight[lightIndex].startingIntensity;
        float lightX = firstLight[lightIndex].lightx;//100.0;
        float lightY = firstLight[lightIndex].lighty;//100.0;
        float radius = firstLight[lightIndex].radius;//300;
        //mapdata[i] -= getLightValue(startingIntensity, lightX, lightY, radius, x, y);
        mapdata[i] -= getLightAmount(x, y, firstLight[lightIndex], obstacles, obstacleCount);
        
        /*
        for (int testIndex = 0; testIndex < 50; testIndex++)
        {
            int index = firstLight[lightIndex].testxpoints[testIndex] + (firstLight[lightIndex].testypoints[testIndex] * 640);
            if ((index >= 0) && (index < (460 * 640))) {
                mapdata[index] = 0.0;
            }
        }
        */
    }

    __syncthreads();
}

__global__ void calc(float* output) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    int y = (int)(i / 640);
    int x = i - (y * 640);

    if (i < 0 || i >(640 * 480)) {
        return;
    }

    output[i * 4] = 34.0;
}

__global__ void forReal(uchar4* dst,light* lights,obstacle* obstacles) {
    const int OBSTACLE_COUNT = 2;

    //int x = blockIdx.x * blockDim.x + threadIdx.x;
    //int y = blockIdx.y * blockDim.y + threadIdx.y;

    //int i = threadIdx.x + blockIdx.x * blockDim.x;

    int i = ((blockDim.x * blockDim.y) * blockIdx.x) + (threadIdx.y * blockDim.x) + threadIdx.x;
    int y = (int) (i / MAP_WIDTH);
    int x = i - (y * MAP_WIDTH);

    if (i < 0 || i > (MAP_WIDTH * MAP_HEIGHT)) {
        return;
    }

    if (x > MAP_WIDTH || y > MAP_HEIGHT) {
        return;
    }

    /*
    float randomPoint1x, randomPoint1y, randomPoint2x, randomPoint2y;
    randomPoint1x = 350.0;
    randomPoint1y = 350.0;
    randomPoint2x = 400.0;
    randomPoint2y = 370.0;
    
    float randomPointRotation = rotation(randomPoint1x, randomPoint1y, randomPoint2x, randomPoint2y);

    if (distance(randomPoint1x, randomPoint1y, x, y) < 5.0) {
        dst[i] = make_uchar4(0.0, 100.0, 0.0, 255.0);
        return;
    }

    if (distance(randomPoint2x, randomPoint2y, x, y) < 5.0) {
        dst[i] = make_uchar4(0.0, 100.0, 0.0, 255.0);
        return;
    }

    if (pointOnLine(randomPoint1x, randomPoint1y, randomPoint2x, randomPoint2y, x, y)) {
        dst[i] = make_uchar4(0.0, 0.0, 100.0, 255.0);
        return;
    }


    float perpPoint1x, perpPoint1y, perpPoint2x, perpPoint2y;
    float perpRotation1, perpRotation2;
    perpRotation1 = randomPointRotation + (0.5 * 3.14);

    if (perpRotation1 > (2 * 3.14)) {
        perpRotation1 -= 2 * 3.14;
    }

    perpRotation2 = perpRotation1 + 3.14;
    if (perpRotation2 > (2 * 3.14)) {
        perpRotation2 -= 2 * 3.14;
    }

    perpPoint1x = randomPoint1x + (30.0 * cos(perpRotation1));
    perpPoint1y = randomPoint1y + (30.0 * sin(perpRotation1));

    if (pointOnLine(randomPoint1x, randomPoint1y, perpPoint1x, perpPoint1y, x, y)) {
        dst[i] = make_uchar4(100.0, 0.0, 100.0, 255.0);
        return;
    }

    perpPoint2x = randomPoint1x + (30.0 * cos(perpRotation2));
    perpPoint2y = randomPoint1y + (30.0 * sin(perpRotation2));

    if (pointOnLine(randomPoint1x, randomPoint1y, perpPoint2x, perpPoint2y, x, y)) {
        dst[i] = make_uchar4(100.0, 0.0, 100.0, 255.0);
        return;
    }


    float lightx, lighty, point1x, point1y, point2x, point2y;
    float testRotation,leftRotation, rightRotation,pct;
    lightx = 150.0;
    lighty = 150.0;
    point1x = 330.9;
    point1y = 330.0;
    point2x = 250.0;
    point2y = 200.0;

    rightRotation = rotation(lightx, lighty, point1x, point1y);
    leftRotation = rotation(lightx, lighty, point2x, point2y);
    testRotation = rotation(lightx, lighty, x, y);
    pct = (testRotation - leftRotation) / (rightRotation - leftRotation);

    float boundaryx, boundaryy,finalDistance;
    boundaryx = ((pct * point1x) + ((1 - pct) * point2x));
    boundaryy = ((pct * point1y) + ((1 - pct) * point2y));
    finalDistance = distance(lightx, lighty, boundaryx, boundaryy);
    //finalDistance = distance(lightx, lighty, point1x, point1y);
    if (testRotation >= leftRotation && testRotation <= rightRotation && distance(lightx,lighty,x,y) < finalDistance) {
        dst[i] = make_uchar4(0.0, 100.0,0.0, 255.0);
        return;
    }


    if (distance(x, y, lightx, lighty) < 3.0) {
        dst[i] = make_uchar4(0.0, 0.0, 100.0, 255.0);
        return;
    }

    if (distance(x, y, point1x, point1y) < 3.0) {
        dst[i] = make_uchar4(0.0, 0.0, 100.0, 255.0);
        return;
    }

    if (distance(x, y, point2x, point2y) < 3.0) {
        dst[i] = make_uchar4(0.0, 0.0, 100.0, 255.0);
        return;
    }


    float tx, ty,radius,firstRotation,secondRotation;
    float rot,dis;
    tx = 100.0;
    ty = 100.0;
    firstRotation = 0.0 * 3.14;
    secondRotation = 0.003 * 3.14;
    dis = distance(tx, ty, x, y);
    ///rotation = atan((y - ty) / (x - tx));
    rot = rotation(tx, ty, x, y);

    if (rot >= firstRotation && rot <= secondRotation) {
        //dst[i] = make_uchar4(0.0, 100.0, 100.0, 255.0);
        //return;
    }
    */

    float Red = 0.0;
    float Green = 0.7 * 255.0;
    float Blue = 0.7 * 255.0;
    float Alpha = 0.91;
    float startingIntensity = 0.0;

    bool withinObstacle = false;

    for (int ob = 0; ob < OBSTACLE_COUNT; ob++) {
        if (withinBox(obstacles[ob].x, obstacles[ob].y, obstacles[ob].width, obstacles[ob].height, x, y)) {
            withinObstacle = true;
            break;
        }
    }
    if (!(withinObstacle)) {
        for (int lightIndex = 0; lightIndex < 100; lightIndex++) {
            if (lights[lightIndex].lightx == -1) {
                break;
            }

            startingIntensity = lights[lightIndex].startingIntensity;
            float lightX = lights[lightIndex].lightx;//100.0;
            float lightY = lights[lightIndex].lighty;//100.0;
            float radius = lights[lightIndex].radius;//300;
            //mapdata[i] -= getLightValue(startingIntensity, lightX, lightY, radius, x, y);
            Alpha -= getLightAmount(x, y, lights[lightIndex], obstacles, OBSTACLE_COUNT);
        }
    }

    

    if (Alpha > 1.0) {
        Alpha = 1.0;
    }

    if (Alpha < 0.0) {
        Alpha = 0.0;
    }
    
    Red -= Red * Alpha;
    Blue -= Blue * Alpha;
    Green -= Green * Alpha;
    
    dst[i] = make_uchar4(Red, Green, Blue, 255.0);
    
    

    //__syncthreads();
}

__global__ void calculateVisibilityTriangles(light* lights, obstacle* obstacles) {

}

void DeleteTexture(GLuint& texture)
{
    if (texture != 0)
    {
        glDeleteTextures(1, &texture);
        texture = 0;
    }
}

void CreateTexture(GLuint& texture, unsigned int width, unsigned int height)
{
    // Make sure we don't already have a texture defined here
    DeleteTexture(texture);

    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    // set basic parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // Create texture data (4-component unsigned byte)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    // Unbind the texture
    glBindTexture(GL_TEXTURE_2D, 0);
}

void DisplayImage(GLuint texture, unsigned int x, unsigned int y, unsigned int width, unsigned int height)
{
    glBindTexture(GL_TEXTURE_2D, texture);
    glEnable(GL_TEXTURE_2D);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glPushAttrib(GL_VIEWPORT_BIT);
    glViewport(x, y, width, height);

    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 0.0); glVertex3f(-1.0, -1.0, 0.5);
    glTexCoord2f(1.0, 0.0); glVertex3f(1.0, -1.0, 0.5);
    glTexCoord2f(1.0, 1.0); glVertex3f(1.0, 1.0, 0.5);
    glTexCoord2f(0.0, 1.0); glVertex3f(-1.0, 1.0, 0.5);
    glEnd();

    glPopAttrib();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    //glDisable(GL_TEXTURE_2D);
}

void drawImage(GLuint file,float x,float y,float w, float h)
{
    //glColor3f(0.0, 100.7, 255.0);
    //glVertex3f(x, y, 0.0f);
    //glVertex3f(x, y + h, 0.0f);
    //glVertex3f(x + w, y + h, 0.0f);
    //glVertex3f(x + w, y, 0.0f);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glPushAttrib(GL_VIEWPORT_BIT);
    glViewport(x, y, MAP_WIDTH, MAP_HEIGHT);

    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);


    glPushMatrix();
    glTranslatef(x, y, 0.0);

    glBindTexture(GL_TEXTURE_2D, file);
    glEnable(GL_TEXTURE_2D);

    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 0.0); glVertex3f(x, y, 0.0f);
    glTexCoord2f(0.0, 1.0); glVertex3f(x, y + h, 0.0f);
    glTexCoord2f(1.0, 1.0); glVertex3f(x + w, y + h, 0.0f);
    glTexCoord2f(1.0, 0.0); glVertex3f(x + w, y, 0.0f);
    glEnd();

    glPopMatrix();
}

void initializeLightsAndObstacles() {
    //Standard
    h_Lights = (light*)malloc(sizeof(light) * LIGHT_COUNT);

    //Unified memory
    //cudaMallocManaged(&h_Lights, LIGHT_COUNT * sizeof(light));

    //Mapped
    //errorCheck(cudaHostAlloc(&h_Lights, sizeof(light) * LIGHT_COUNT, cudaHostAllocMapped));
    //errorCheck(cudaHostGetDevicePointer(&d_Lights, h_Lights, 0));

    for (int i = 0; i < LIGHT_COUNT; i++) {
        h_Lights[i].lightx = -1;
        h_Lights[i].lighty = -1;
        h_Lights[i].radius = -1;
        h_Lights[i].startingIntensity = -1;
    }

    h_Lights[0].lightx = 100;
    h_Lights[0].lighty = 100;
    h_Lights[0].radius = 300;
    h_Lights[0].startingIntensity = 0.8;

    h_Lights[1].lightx = 150;
    h_Lights[1].lighty = 300;
    h_Lights[1].radius = 300;
    h_Lights[1].startingIntensity = 0.8;

    //for (int randomIndex = 0; randomIndex < LIGHT_SAMPLE_POINTS; randomIndex++) {
    //    float rotation = ((float)randomIndex / (float)LIGHT_SAMPLE_POINTS) * 2.0 * 3.14;
    //    h_Lights[0].testxpoints[randomIndex] = h_Lights[0].lightx + LIGHT_SAMPLE_POINT_RADIUS * cos(rotation);
    //    h_Lights[0].testypoints[randomIndex] = h_Lights[0].lighty + LIGHT_SAMPLE_POINT_RADIUS * sin(rotation);
    //}

    h_obstacles = (obstacle*)malloc(sizeof(obstacle) * OBSTACLE_COUNT);

    h_obstacles[0].x = 200;
    h_obstacles[0].y = 200;
    h_obstacles[0].width = 25;
    h_obstacles[0].height = 25;

    h_obstacles[1].x = 325;
    h_obstacles[1].x = 325;
    h_obstacles[1].y = 200;
    h_obstacles[1].width = 25;
    h_obstacles[1].height = 25;

    errorCheck(cudaMalloc((light**)&d_Lights, sizeof(light) * LIGHT_COUNT));
    errorCheck(cudaMallocHost((void**)&d_Lights, sizeof(light) * LIGHT_COUNT));
    errorCheck(cudaMemcpy(d_Lights, h_Lights, sizeof(light) * LIGHT_COUNT, cudaMemcpyHostToDevice));

    errorCheck(cudaMalloc((obstacle**)&d_obstacles, sizeof(obstacle) * OBSTACLE_COUNT));
    errorCheck(cudaMemcpy(d_obstacles, h_obstacles, sizeof(obstacle) * OBSTACLE_COUNT, cudaMemcpyHostToDevice));
}

void initializeOpenGL()
{
    /* Initialize the library */
    if (!glfwInit())
        return;

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(MAP_WIDTH, MAP_HEIGHT, "Light Sim", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return;
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    //glfwGetFramebufferSize(window, MAP_WIDTH, MAP_HEIGHT);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    //glOrtho(0, MAP_WIDTH, MAP_HEIGHT, 0, -1, 1);
    glOrtho(0, MAP_WIDTH, MAP_HEIGHT, 0, 0, 1);

    glewInit();
}

float distanceH(float x1, float y1, float x2, float y2) {
    float xdif = x1 - x2;
    float ydif = y1 - y2;
    //return std::sqrt((xdif * xdif) + (ydif * ydif));
    return sqrt((xdif * xdif) + (ydif * ydif));
}

inline bool withinBoxH(float bx, float by, float bw, float bh, float tx, float ty) {
    return ((tx > bx) && (tx < (bx + bw)) && (ty > by) && (ty < (by + bh)));
}

bool rayTraceH(float x1, float y1, float x2, float y2, obstacle* obstacleList, int obstacleCount) {
    float pointsPerDistance = 0.3;
    int points = (int)(distanceH(x1, y1, x2, y2) * pointsPerDistance);
    float xstep, ystep;
    float cx, cy;

    xstep = (x2 - x1) / ((float)points);
    ystep = (y2 - y1) / ((float)points);

    cx = x1;
    cy = y1;

    for (int i = 0; i < points; i++) {
        for (int obstacleIndex = 0; obstacleIndex < obstacleCount; obstacleIndex++) {
            if (withinBoxH(obstacleList[obstacleIndex].x, obstacleList[obstacleIndex].y, obstacleList[obstacleIndex].width, obstacleList[obstacleIndex].height, cx, cy)) {
                return false;
            }

            cx += xstep;
            cy += ystep;
        }
    }

    return true;
}



float rotationH(float x1, float y1, float x2, float y2) {
    float xdif, ydif;
    xdif = x2 - x1;
    ydif = y2 - y1;

    float rot = atan(abs(ydif) / abs(xdif));

    if (xdif >= 0.0) {
        if (ydif >= 0.0) {
            return rot;
        }
        else {
            return (2 * 3.14) - rot;
        }
    }
    else {
        if (ydif >= 0.0) {
            return 3.14 - rot;
        }
        else {
            return 3.14 + rot;
        }
    }
}



void calculateVisibilityTrianglesAA(light* lights, obstacle* obstacles) {
    float lastLength = 0.0;
    float currentLength = 0.0;
    float currentRotation = 0.0;
    float rotationIncrement = (2 * 3.14) / 100.0;
    float perpRotation1, perpRotation2;
    float perp1x, perp1y, perp2x, perp2y;
    float traceX, traceY;
    float testX, testY;

    for (int lightIndex = 0; lightIndex < LIGHT_COUNT; lightIndex++) {
        lastLength = 0.0;
        currentLength = 0.0;
        currentRotation = 0.0;


        for (int i = -1; i < 100; i++) {
            perpRotation1 = currentRotation + (3.14 * 0.5);
            if (perpRotation1 > (2 * 3.14)) {
                perpRotation1 -= (2 * 3.14);
            }

            perpRotation2 = perpRotation1 + 3.14;
            if (perpRotation2 > (2 * 3.14)) {
                perpRotation2 -= (2 * 3.14);
            }

            perp1x = lights[lightIndex].lightx + (15.0 * cos(perpRotation1));
            perp1y = lights[lightIndex].lighty + (15.0 * sin(perpRotation1));
            perp2x = lights[lightIndex].lightx + (15.0 * cos(perpRotation2));
            perp2y = lights[lightIndex].lighty + (15.0 * sin(perpRotation2));

            traceX = 10.0 * cos(currentRotation);
            traceY = 10.0 * sin(currentRotation);

            testX = lights[lightIndex].lightx + traceX;
            testY = lights[lightIndex].lighty + traceY;

            currentLength = 0.0;
            while (currentLength < 280.0) {
                if (!(rayTraceH(perp1x, perp1y, testX, testY, obstacles, OBSTACLE_COUNT))) {
                    break;
                }

                if (!(rayTraceH(perp2x, perp2y, testX, testY, obstacles, OBSTACLE_COUNT))) {
                    break;
                }

                currentLength += 30.0;
                testX += traceX;
                testY += traceY;
            }

            if (!(i == -1)) {
                //lights[lightIndex].visibilityLeftRotations[i] = currentRotation - rotationIncrement;
                //lights[lightIndex].visibilityRightRotations[i] = currentRotation - rotationIncrement;

                if (lastLength <= currentLength)
                {
                    //lights[lightIndex].visibilityRadius[i] = lastLength;
                }
                else
                {
                    //lights[lightIndex].visibilityRadius[i] = currentLength;
                }
            }
            lastLength = currentLength;
        }
    }
}

int main()
{
    initializeOpenGL();
    initializeLightsAndObstacles();

    CreateTexture(lightMapTexture, MAP_WIDTH, MAP_HEIGHT);
    cudaGraphicsGLRegisterImage(&lightMapTextureResource, lightMapTexture, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard);

    std::chrono::time_point<std::chrono::system_clock> lastFrameOutput,currentTime;
    std::chrono::duration<double>  elapsed;
    int frames = 0;

    lastFrameOutput = std::chrono::system_clock::now();

    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
        /* Render here */
        //glColor3f(0.0, 0.7, 0.7);
        //glClear(GL_COLOR_BUFFER_BIT);
        
        //glClearColor(0.0, 0.7, 0.7, 1.0);
        int window_width, window_height;
        
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        //glfwGetFramebufferSize(window, MAP_WIDTH, MAP_HEIGHT);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        //glOrtho(0, MAP_WIDTH, MAP_HEIGHT, 0, -1, 1);
        glOrtho(0, MAP_WIDTH, MAP_HEIGHT, 0, 0, 1);

        h_Lights[0].lightx += 5.0;
        h_Lights[1].lightx += 5.0;

        //calculateVisibilityTriangles(h_Lights, h_obstacles);
        // 
        //for (int randomIndex = 0; randomIndex < LIGHT_SAMPLE_POINTS; randomIndex++) {
         ///   float rotation = ((float)randomIndex / (float)LIGHT_SAMPLE_POINTS) * 2.0 * 3.14;
        //    h_Lights[0].testxpoints[randomIndex] = h_Lights[0].lightx + LIGHT_SAMPLE_POINT_RADIUS * cos(rotation);
        //    h_Lights[0].testypoints[randomIndex] = h_Lights[0].lighty + LIGHT_SAMPLE_POINT_RADIUS * sin(rotation);
        //}

        std::chrono::time_point<std::chrono::system_clock> copyStart, copyEnd;
        std::chrono::duration<double> copyTime;
        copyStart = std::chrono::system_clock::now();

        errorCheck(cudaMemcpy(d_Lights, h_Lights, sizeof(light) * LIGHT_COUNT, cudaMemcpyHostToDevice));

        copyEnd = std::chrono::system_clock::now();
        copyTime = copyEnd - copyStart;
        //std::cout << "Copy Time: " << copyTime.count() << "s\n";


        cudaGraphicsResource_t resources[1] = { lightMapTextureResource };
        cudaGraphicsMapResources(1, resources);
        cudaArray* dstArray;
        cudaGraphicsSubResourceGetMappedArray(&dstArray, lightMapTextureResource, 0, 0);
        //cudaBindTextureToArray(texRef, srcArray);

        

        size_t bufferSize = MAP_WIDTH * MAP_HEIGHT * sizeof(uchar4);
        if (g_BufferSize != bufferSize)
        {
            if (g_dstBuffer != NULL)
            {
                cudaFree(g_dstBuffer);
            }
            // Only re-allocate the global memory buffer if the screen size changes, 
            // or it has never been allocated before (g_BufferSize is still 0)
            g_BufferSize = bufferSize;
            cudaMalloc(&g_dstBuffer, g_BufferSize);
        }

        

        std::chrono::time_point<std::chrono::system_clock> kernelstart,kernelend;
        std::chrono::duration<double> kernelTime;
        kernelstart = std::chrono::system_clock::now();


        dim3 threadsPerBlock(8,8);
        //dim3 numBlocks((MAP_WIDTH * MAP_HEIGHT) / threadsPerBlock.x, (MAP_WIDTH * MAP_HEIGHT) / threadsPerBlock.y);
        //MatAdd << <numBlocks, threadsPerBlock >> > (A, B, C);

        //calculateVisibilityTriangles << <1, 200 >> > (d_Lights, d_obstacles);


        //forReal << <615, 500 >> > (g_dstBuffer,d_Lights,d_obstacles);
        int numBlocks = (MAP_WIDTH * MAP_HEIGHT) / (8 * 8);
        numBlocks += 5;
        forReal << <numBlocks, threadsPerBlock >> > (g_dstBuffer, d_Lights, d_obstacles);
        //cudaDeviceSynchronize();
        kernelend = std::chrono::system_clock::now();
        kernelTime = kernelend - kernelstart;
        //std::cout << "Kernel Time: " << kernelTime.count() << "s\n";

        

        cudaMemcpyToArray(dstArray, 0, 0, g_dstBuffer, bufferSize, cudaMemcpyDeviceToDevice);
        cudaGraphicsUnmapResources(1, resources);
        drawImage(lightMapTexture, 0, 0, MAP_WIDTH, MAP_HEIGHT);

        

        //DisplayImage(lightMapTexture, 0, 0, MAP_WIDTH, MAP_HEIGHT);

        processKey(window);
        glfwSwapBuffers(window);
        glfwPollEvents();

        frames++;
        currentTime = std::chrono::system_clock::now();
        elapsed = currentTime - lastFrameOutput;
        if (elapsed.count() > 1.0) {
            std::cout << "Frame Rate: " << frames << "\n";
            lastFrameOutput = std::chrono::system_clock::now();
            frames = 0;
        }

        //std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    glfwTerminate();
    return 0;
}

void processKey(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        for (int x = 50; x < 75; x++) {
            for (int y = 50; y < 75; y++) {
                glColor3f(0.0, 0.7, 0.7);
                glVertex2i(x, y);
            }
        }
    }
        
}



void errorCheck(int errorValue) {
    if (!(errorValue == 0)) {
        std::cout << "Error." << std::endl;
    }
}

void endProgram() {
    int a = 0;
    std::cout << "Press any key to continue." << "\n";
    std::cin >> a;
    exit(0);
}

