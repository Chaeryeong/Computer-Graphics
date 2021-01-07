#include "scene.h"

#include "obj_teapot.h"
#include "tex_flower.h"

Shader* Scene::vertexShader = nullptr;
Shader* Scene::fragmentShader = nullptr;
Program* Scene::program = nullptr;
Camera* Scene::camera = nullptr;
Light* Scene::light = nullptr;
Object* Scene::teapot = nullptr;
Material* Scene::flower = nullptr;

void Scene::setup(AAssetManager* aAssetManager) {

    // set asset manager
    Asset::setManager(aAssetManager);

    // create shaders
    vertexShader = new Shader(GL_VERTEX_SHADER, "vertex.glsl");
    fragmentShader = new Shader(GL_FRAGMENT_SHADER, "fragment.glsl");

    // create program
    program = new Program(vertexShader, fragmentShader);

    // create camera
    camera = new Camera(program);
    camera->eye = vec3(60.0f, 00.0f, 0.0f);

    // create light
    light = new Light(program);
    light->position = vec3(100.0f, 0.0f, 0.0f);

    // create floral texture
    flower = new Material(program, texFlowerData, texFlowerSize);

    // create teapot object
    teapot = new Object(program, flower, objTeapotVertices, objTeapotIndices,
                        objTeapotVerticesSize, objTeapotIndicesSize);

    // Problem #2
    mat4 eulerX, eulerY, scaleM;
    eulerX = transpose(mat4(1.0f, 0.0f, 0.0f, 0.0f,  //In OpenGL, the matrix must be transposed
                            0.0f, cos(45), -sin(45), 0.0f,
                            0.0f, sin(45), cos(45), 0.0f,
                            0.0f, 0.0f, 0.0f, 1.0f));

    eulerY = transpose(mat4(cos(45), 0.0f, sin(45), 0.0f,
                            0.0f, 1.0f, 0.0f, 0.0f,
                            -sin(45), 0.0f, cos(45), 0.0f,
                            0.0f, 0.0f, 0.0f, 1.0f));

    scaleM = transpose(mat4(1.0f, 0.0f, 0.0f, 0.0f,
                            0.0f, 1.5f, 0.0f, 0.0f,
                            0.0f, 0.0f, 0.5f, 0.0f,
                            0.0f, 0.0f, 0.0f, 1.0f));
    teapot->worldMatrix = eulerY*eulerX*scaleM*teapot->worldMatrix;
}

void Scene::screen(int width, int height) {

    // set camera aspect ratio
    camera->aspect = (float) width / height;
}

void Scene::update(float deltaTime) {
    static float angle = 0.0f;

    // use program
    program->use();

    // Problem #3
    // rotate the camera relative to the object
    mat3 camRotate;
    camRotate = transpose(mat3(cos(-deltaTime), 0.0f, sin(-deltaTime),
                               0.0f, 1.0f, 0.0f,
                               -sin(-deltaTime), 0.0f, cos(-deltaTime)));
    camera->eye = camRotate*camera->eye;

    // setup camera and light
    camera->setup();
    light->setup();

    // draw teapot
    teapot->draw();
}

