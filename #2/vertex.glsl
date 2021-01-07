#version 300 es

uniform mat4 worldMat, viewMat, projMat;
uniform vec3 eyePos;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texCoord;
layout(location = 3) in vec3 tangent;

out vec3 v_normal;
out vec2 v_texCoord;
out vec3 v_view;

void main() {
    //┌ 3. Implement Phong lighting────────────────────────────┐
        vec3 worldPos = (worldMat * vec4(position, 1.0)).xyz;
        v_view = normalize(eyePos - worldPos);

    //└──────────────────────────────────────────┘

    v_normal = normalize(transpose(inverse(mat3(worldMat))) * normal);
    v_texCoord = texCoord;
    gl_Position = projMat * viewMat * worldMat * vec4(position, 1.0);
}