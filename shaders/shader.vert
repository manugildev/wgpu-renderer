// shader.vert
#version 450

layout(location=0) in vec3 a_position;
layout(location=1) in vec2 a_tex_coords;
layout(location=2) in vec3 a_normal;

layout(location=0) out vec2 v_tex_coords;
layout(location=1) out vec3 v_normal;
layout(location=2) out vec3 v_position;

layout(set=1, binding=0) uniform Uniforms {
    vec3 u_view_position; // unused
    mat4 u_view_proj;
};

layout(location=5) in mat4 model_matrix;


void main() {
    // TODO: Calculate normal matrix on CPU and pass it (is less expensive)
    mat3 normal_matrix = mat3(transpose(inverse(model_matrix)));
    v_normal = normal_matrix * a_normal;

    vec4 model_space = model_matrix * vec4(a_position, 1.0);
    gl_Position = u_view_proj * model_space;

    // Pass data to fragment shader
    v_position = model_space.xyz;
    v_tex_coords = a_tex_coords;
    v_normal = a_normal;
}

