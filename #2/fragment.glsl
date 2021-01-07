#version 300 es

precision mediump float;

uniform sampler2D textureDiff;
uniform sampler2D textureDissolve;

uniform vec3 matSpec, matAmbi, matEmit;
uniform float matSh;
uniform vec3 srcDiff, srcSpec, srcAmbi;
uniform float threshold;
uniform vec3 lightPos;

in vec3 v_normal, v_view;
in vec2 v_texCoord;

layout(location = 0) out vec4 fragColor;

void main() {
     vec3 color = texture(textureDiff, v_texCoord).rgb;

    //┌ 3. Implement Phong lighting────────────────────────────┐

        // normalization

        vec3 normal = normalize(v_normal);
        vec3 view = normalize(v_view);
        vec3 light = normalize(lightPos);

        // diffuse term
        vec3 matDiff = texture(textureDiff, v_texCoord).rgb;
        vec3 diff = max(dot(normal, light), 0.0) * srcDiff * matDiff;

        // specular term
        vec3 refl = 2.0 * normal * dot(normal, light) - light;
        vec3 spec = pow(max(dot(refl, view), 0.0), matSh) * srcSpec * matSpec;

        // ambient term
        vec3 ambi = srcAmbi * matAmbi;

        // sum up
        color = vec3(diff + spec + ambi + matEmit);

    //└──────────────────────────────────────────┘


    float alpha = 1.0f;
    //┌ 4. Implement alpha blending────────────────────────────┐
       float dissolve = texture(textureDissolve, v_texCoord).r;;
       if (threshold>dissolve){
          alpha = 0.0f; }
    //└──────────────────────────────────────────┘


    // final output color with alpha
    fragColor = vec4(color, alpha);
}