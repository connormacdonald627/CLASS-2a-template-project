#version 330

in mat4 instance_model;
in vec3 in_position;
in vec3 in_colour;
in vec3 instance_color;
in vec2 in_uv;
in vec3 in_normal;

uniform vec4 colour = vec4(1.0, 1.0, 1.0, 1.0);
uniform vec4 tiling_offset = vec4(1.0, 1.0, 0.0, 0.0);
uniform int is_instanced;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 world_position;
out vec3 vertex_colour;
out vec3 normal;
out vec3 world_normal;
out vec2 uv;
out vec4 frag_colour;

void main() {
    bool is_instance = is_instanced > 0;
    mat4 current_model = is_instance ? instance_model : model;
    gl_Position = vec4(in_position, 1.0) * current_model * view * projection;

    vertex_colour = is_instance ? instance_color.rgb : in_colour;
    frag_colour = colour;
    normal = in_normal;
    mat3 normal_matrix = transpose(inverse(mat3(current_model)));
    world_normal = normalize(normal_matrix * in_normal);
    world_position = (vec4(in_position, 1.0) * current_model).xyz;
    uv = in_uv * tiling_offset.xy + tiling_offset.zw;
}