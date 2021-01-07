#include "scene.h"
#include "binary/teapot.h"
#include "binary/floral.h"
#include "binary/cloud.h"
#include "checker.h"

Shader* Scene::vertexShader = nullptr;
Shader* Scene::fragmentShader = nullptr;
Program* Scene::program = nullptr;
Camera* Scene::camera = nullptr;
Object* Scene::teapot = nullptr;
Texture* Scene::diffuse = nullptr;
Texture* Scene::dissolve = nullptr;
Material* Scene::material = nullptr;
Light* Scene::light = nullptr;

void Scene::setup(AAssetManager* aAssetManager) {
    Asset::setManager(aAssetManager);

    Scene::vertexShader = new Shader(GL_VERTEX_SHADER, "vertex.glsl");
    Scene::fragmentShader = new Shader(GL_FRAGMENT_SHADER, "fragment.glsl");

    Scene::program = new Program(Scene::vertexShader, Scene::fragmentShader);

    Scene::camera = new Camera(Scene::program);
    Scene::camera->eye = vec3(10.0f, 20.0f, 10.0f);

    Scene::light = new Light(program);
    light->direction = vec3(-3.0f, 9.0f, 9.0f);

    Scene::diffuse  = new Texture(Scene::program, 0, "textureDiff", floralTexels, floralSize);
    //┌ 2. Change the texture wrapping ──────────────────────────┐
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    //└──────────────────────────────────────────┘
    Scene::dissolve = new Texture(Scene::program, 1, "textureDissolve", cloudTexels, cloudSize);
    Scene::material = new Material(Scene::program, diffuse, dissolve);
    Scene::teapot = new Object(program, material, teapotVertices, teapotIndices);
}

void Scene::screen(int width, int height) {
    Scene::camera->aspect = (float) width/height;
}
void Scene::update(float deltaTime) {

    Scene::program->use();

    //┌ 4. Implement alpha blending────────────────────────────┐
       Scene::teapot->material->threshold += deltaTime/10.0f;
    //└──────────────────────────────────────────┘

    Scene::camera->update();
    Scene::light->update();

    Scene::teapot->draw();
}