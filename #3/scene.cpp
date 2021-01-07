#include "scene.h"

#include "obj_teapot.h"
#include "tex_flower.h"

#define NUM 3



Shader* Scene::vertexShader = nullptr;
Shader* Scene::fragmentShader = nullptr;
Program* Scene::program = nullptr;
Camera* Scene::camera = nullptr;
Light* Scene::light = nullptr;
Object** Scene::teapot = nullptr;
Material* Scene::flower = nullptr;
int Scene::_width = 0;
int Scene::_height = 0;

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
    camera->eye = vec3(0.0f, 00.0f, 60.0f);

    // create light
    light = new Light(program);
    light->position = vec3(0.0f, 0.0f, 100.0f);

    // create floral texture
    flower = new Material(program, texFlowerData, texFlowerSize);

    // create teapot object
    teapot = new Object*[NUM];

    for (int i = 0; i < NUM; i++)
    {
        teapot[i] = new Object(program, flower, objTeapotVertices, objTeapotIndices,
                               objTeapotVerticesSize, objTeapotIndicesSize);
        teapot[i]->origin = vec3(0.0f, 0.0f+ (i*20.0f), 0.0f - (i*20.0f));
        teapot[i]->worldMatrix = translate(teapot[i]->origin) * rotate(radians(60.0f), vec3(1.0, 1.0, 0.0));
    }

    int i = 0;
    //LOG_PRINT_DEBUG("Hello World %d %d\n", i, i+1);
}

void Scene::mouseUpEvents(float x, float y, bool doubleTouch)
{
}

void Scene::screen(int width, int height) {

    _width = width;
    _height = height;

    // set camera aspect ratio
    camera->aspect = (float) width / height;
}

void Scene::update(float deltaTime) {

    // use program
    program->use();

    // setup camera and light
    camera->setup();
    light->setup();

    // draw teapot
    for (int i = 0; i < NUM; i++)
    {
        teapot[i]->origin = vec3(teapot[i]->worldMatrix[3][0],
                teapot[i]->worldMatrix[3][1], teapot[i]->worldMatrix[3][2]);
        teapot[i]->draw();
    }
}


//┌───────────────────────────────────────────────────────────┐
int Scene::pickIndex = -1;
vec3 Scene::oldVec = vec3(0.0f, 0.0f, 0.0f);

vec4 Scene::inverseTransform(float x, float y, bool isPoint)
{
    /*
    return the ray component which is in WORLD space
        - parameter x, y: screen space coordinates (Xs, Ys)
        - isPoint == (true/false), return the (start point/direction vector) - vec4
     */
    /*
     screen S -> clip S -> camera S -> world S
        # step 1: screen S to camera S
        # step 2: camera S to world S
        # step 3: store to oldVec & return vec4 depending on whether it's point or vector
     */
    // step#1: Screen Space to Camera Space
    mat4x4 viewportMatrix; //proj: cam->clip, viewport: clip->screen

    float w = (float)_width;
    float h = (float)_height;
    viewportMatrix = transpose(mat4x4((float)(w/2), 0.0f, 0.0f, (float)(w/2),
                                    0.0f, -(float)(h/2), 0.0f, (float)(h/2),
                                   0.0f, 0.0f, 0.5f, 0.5f,
                                   0.0f, 0.0f, 0.0f, 1.0f));

    vec4 screen_coord = vec4(x, y, 0.0f, 1.0f);

    vec4 clip_coord = inverse(viewportMatrix)*screen_coord;
    vec4 camera_coord = inverse(camera->projMatrix)*clip_coord;
    vec4 camera_dirVec = camera_coord - vec4(0.0f, 0.0f, 0.0f, 1.0f);
    //LOG_PRINT_DEBUG("cam before x y z w : %f %f %f  %f\n", camera_dirVec.x, camera_dirVec.y, camera_dirVec.z, camera_dirVec.w);
    //camera_dirVec = camera_dirVec/camera_dirVec.z;
    //LOG_PRINT_DEBUG("cam after x y z w : %f %f %f  %f\n", camera_dirVec.x, camera_dirVec.y, camera_dirVec.z, camera_dirVec.w);

    // step#2: Camera Space to World Space
    vec4 world_coord = inverse(camera->viewMatrix)*camera_coord;
    vec4 world_dirVec = inverse(camera->viewMatrix)*camera_dirVec;

    // step#3: return vec4
    if(isPoint){
        return world_coord;
    }
    else{ //isPoint==false
        return world_dirVec;
    }
}

vec3 Scene::calculateArcballvec(float x, float y)
{
    /*
     return the vector v connection the arcball's center and the projected line
     */
    mat4x4 viewportMatrix = transpose(mat4x4((float)(_width/2), 0.0f, 0.0f, (float)(_width/2),
                                      0.0f, -(float)(_height/2), 0.0f, (float)(_height/2),
                                      0.0f, 0.0f, 0.5f, 0.5f,
                                      0.0f, 0.0f, 0.0f, 1.0f));

    vec4 temp = inverse(viewportMatrix)*vec4(x, y, 0.0f, 1.0f);

    vec3 v;
    float qx = temp.x;
    float qy = temp.y;

    /*
     handling the problem of teapot disappearing when the mouse out of the screen
     -> clam into the NDC [-1,1]
     */
    qx = (qx>1)? 1:qx;
    qx = (qx<-1)? -1:qx;
    qy = (qy>1)? 1:qy;
    qy = (qy<-1)? -1:qy;

    if((pow(qx,2)+pow(qy,2))>1){
        v = normalize(vec3(qx, qy, 0.0f));
        // due to the type float, it can be slightly larger than 1 or less than 0 even after the normalization
        //LOG_PRINT_DEBUG("after normalization x y : %f  %f\n", v.x, v.y);
        if(v.y>0){v.y-=0.001;}
        if(v.y<0){v.y+=0.001;}
    }else{
        v = vec3(qx, qy, sqrt(1-pow(qx,2)-pow(qy, 2)));
    }

    return v;
}

void Scene::mouseDownEvents(float x, float y, bool doubleTouch)
{
    /* called automatically
     perform the ray-intersection using a sphere BV
        - parameter x, y: screen space coordinates (Xs,Ys) of the mouse pointer
        - ray- intersection is performed only when doubleTouch==false (i.e. single click)
        - ray component - obtained using the inverseTransform method(returns ray component of world space)
                        - must be converted into the object space
        - 3 objects => t-value comparision should be done
     */
    if(doubleTouch==false){
        pickIndex = -1;
        // single click -> perform the ray-intersection using a sphere BV
        // step#1: get a ray component from the inverseTransform method & covert it into the object space
        vec4 obj1_dirVec = vec4(inverse(mat4x4(teapot[0]->worldMatrix))*inverseTransform(x,y,false));
        vec4 obj1_startP = vec4(inverse(mat4x4(teapot[0]->worldMatrix))*inverseTransform(x,y,true));
        //LOG_PRINT_DEBUG("Vec_obj %f %f %f\n", obj1_startP.x, obj1_startP.y, obj1_dirVec.z);

        vec4 obj2_dirVec = vec4(inverse(mat4x4(teapot[1]->worldMatrix))*inverseTransform(x,y,false));
        vec4 obj2_startP = vec4(inverse(mat4x4(teapot[1]->worldMatrix))*inverseTransform(x,y,true));

        vec4 obj3_dirVec = vec4(inverse(mat4x4(teapot[2]->worldMatrix))*inverseTransform(x,y,false));
        vec4 obj3_startP = vec4(inverse(mat4x4(teapot[2]->worldMatrix))*inverseTransform(x,y,true));

        // step#2: perform the ray-intersection using a sphere BV of each object
            // for obj#1
        double dx_1 = obj1_dirVec.x;
        double dy_1 = obj1_dirVec.y;
        double dz_1 = obj1_dirVec.z;
        double sx_1 = obj1_startP.x;
        double sy_1 = obj1_startP.y;
        double sz_1 = obj1_startP.z;
        int r_1 = 49; // for convenience, r_1 represents the r*r of the sphere BV of the object 1, where r=7

        double a_1 = pow(dx_1, 2) + pow(dy_1, 2) + pow(dz_1, 2);
        double b_1 = 2*sx_1*dx_1 + 2*sy_1*dy_1 + 2*sz_1*dz_1;
        double c_1 = pow(sx_1, 2) + pow(sy_1, 2) + pow(sz_1, 2) - r_1;
        double d_1 = pow(b_1, 2) - 4*a_1*c_1;

        int num_1 = 1;
        if(d_1<0){num_1=0;}

        double t_1 = 0;
        t_1 = (double)((-b_1-sqrt(d_1))/2*a_1);


            // for obj#2
        double dx_2 = obj2_dirVec.x;
        double dy_2 = obj2_dirVec.y;
        double dz_2 = obj2_dirVec.z;
        double sx_2 = obj2_startP.x;
        double sy_2 = obj2_startP.y;
        double sz_2 = obj2_startP.z;
        int r_2 = 49; // for convenience, r_2 represents the r*r of the sphere BV of the object 2, where r=7

        double a_2 = pow(dx_2, 2) + pow(dy_2, 2) + pow(dz_2, 2);
        double b_2 = 2*sx_2*dx_2 + 2*sy_2*dy_2 + 2*sz_2*dz_2;
        double c_2 = pow(sx_2, 2) + pow(sy_2, 2) + pow(sz_2, 2) - r_2;
        double d_2 = pow(b_2, 2) - 4*a_2*c_2;

        int num_2 = 1;
        if(d_2<0){num_2=0;}

        double t_2 = 0;
        t_2 = (double)((-b_2-sqrt(d_2))/2*a_2);


            // for obj#3
        double dx_3 = obj3_dirVec.x;
        double dy_3 = obj3_dirVec.y;
        double dz_3 = obj3_dirVec.z;
        double sx_3 = obj3_startP.x;
        double sy_3 = obj3_startP.y;
        double sz_3 = obj3_startP.z;
        int r_3 = 49; // for convenience, r_3 represents the r*r of the sphere BV of the object 3, where r = 7

        double a_3 = pow(dx_3, 2) + pow(dy_3, 2) + pow(dz_3, 2);
        double b_3 = 2*sx_3*dx_3 + 2*sy_3*dy_3 + 2*sz_3*dz_3;
        double c_3 = pow(sx_3, 2) + pow(sy_3, 2) + pow(sz_3, 2) - r_3;
        double d_3 = pow(b_3, 2) - 4*a_3*c_3;

        int num_3 = 1;
        if(d_3<0){num_3=0;}

        double t_3 = 0;
        t_3 = (double)((-b_3-sqrt(d_3))/2*a_3);

        //LOG_PRINT_DEBUG("a1 a2 a3 %f %f %f\n", a_1, a_2, a_3);
        //LOG_PRINT_DEBUG("b1 b2 b3 %f %f %f\n", b_1, b_2, b_3);
        //LOG_PRINT_DEBUG("c1 c2 c3 %f %f %f\n", c_1, c_2, c_3);
        //LOG_PRINT_DEBUG("d1 d2 d3 %f %f %f\n", d_1, d_2, d_3);
        //LOG_PRINT_DEBUG("t1 t2 t3 %f %f %f\n", t_1, t_2, t_3);

        // step#3: do a t-value comparision and store the index of the object into the pickIndex
        if(num_1>0 && num_2>0 && num_3>0){
            double min = t_1;
            pickIndex = 0;
            if(min>t_2){
                min = t_2;
                pickIndex = 1;
            }
            if(min>t_3){
                min = t_3;
                pickIndex = 2;
            }
        }else if(num_1>0 && num_2>0 && num_3==0){
            double min = t_1;
            pickIndex = 0;
            if(min>t_2){
                min = t_2;
                pickIndex = 1;
            }
        }else if(num_1>0 && num_2==0 && num_3>0){
            double min = t_1;
            pickIndex = 0;
            if(min>t_3){
                min = t_3;
                pickIndex = 2;
            }
        }else if(num_1==0 && num_2>0 && num_3>0){
            double min = t_2;
            pickIndex = 1;
            if(min>t_3){
                min = t_3;
                pickIndex = 2;
            }
        }else if(num_1>0 && num_2==0 && num_3==0){
            pickIndex = 0;
        }else if(num_1==0 && num_2>0 && num_3==0){
            pickIndex = 1;
        }else if(num_1==0 && num_2==0 && num_3>0){
            pickIndex = 2;
        }else{
            pickIndex = -1;
        }
        // 여기서 oldVec값 지정
        //oldVec = vec3(x, y, 0.0f); // here -> false로 바꿔볼까...!
        // screen 으로 설정할까..?
        oldVec = inverseTransform(x, y, true);
    }
    //LOG_PRINT_DEBUG("object picked: %d\n", pickIndex);

    if(doubleTouch){
        oldVec = calculateArcballvec(x, y);
    }
}


void Scene::mouseMoveEvents(float x, float y, bool doubleTouch)
{
    /* called automatically
        doubleTouch == false
        ->implement the selected object(pickIndex) tracks the mouse pointer
        - easily implemented using inverseTransform method(returns ray component of world space)
        - parameter x, y: screen space coordinates (Xs,Ys) of the mouse pointer
     */
        // translate the object of the pickIndex
        // step#1: compute the translation of x-coord, y-coord, z: doesn't change
        if((!doubleTouch) && (pickIndex!=-1)){
            vec3 from = oldVec;
            vec3 to = vec3(inverseTransform(x,y,true));
            // using the resemblence of the triangels of (0,0,60), from, to, z values of each teapot
                // 60-from.z : 60-teapot[]->origin.z = (x,y): x', y')
            float x_trans = to.x - from.x;
            float y_trans = to.y - from.y;

            float real_x_trans = x_trans*(60-(teapot[pickIndex]->origin.z))/(60-from.z);
            float real_y_trans = y_trans*(60-(teapot[pickIndex]->origin.z))/(60-from.z);

            //mat4x4 transMatrix = transpose(mat4x4(1.0f, 0.0f, 0.0f, real_x_trans,
            //                                      0.0f, 1.0f, 0.0f, real_y_trans,
            //                                      0.0f, 0.0f, 1.0f, 0.0f,
            //                                      0.0f, 0.0f, 0.0f, 1.0f));
            //teapot[pickIndex]->worldMatrix = transMatrix*(teapot[pickIndex]->worldMatrix);
            teapot[pickIndex]->worldMatrix = translate(vec3(real_x_trans, real_y_trans, 0))*(teapot[pickIndex]->worldMatrix);
            // oldVec update
            oldVec = to;
        }
        if(doubleTouch && (pickIndex!=-1)){
            vec3 v_nxt = calculateArcballvec(x, y); //new

            vec3 axis = cross(oldVec, v_nxt);


            //  두 벡터 같으면 if(axis==nan()){return;}
            if(v_nxt == oldVec){return;}

            // 두 벡터 평행하면 return
            if(axis ==  vec3(0,0,0)){return;}

            // axis & angle valid할 때만 업뎃하도록
            float radian = acos(dot(oldVec, v_nxt));
            if(radian == 0){return;}

            //from cameraSpace to objectSpace
            vec4 axis_obj = inverse(teapot[pickIndex]->worldMatrix)*inverse(camera->viewMatrix)*vec4(axis,0.0f);
            teapot[pickIndex]->worldMatrix = (teapot[pickIndex]->worldMatrix)*rotate(radian, vec3(axis_obj)); //vec4->vec3로 scaling됨?
            //LOG_PRINT_DEBUG("angle : %f, axis : (%f, %f, %f)", radian, axis_obj.x, axis_obj.y, axis_obj.z);

            oldVec = v_nxt;
        }
}
//└───────────────────────────────────────────────────────────┘









