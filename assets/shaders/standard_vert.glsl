#version 330
in vec3 in_position;
in vec3 in_colour;
in vec2 in_uv;
in vec3 in_normal;

uniform vec4 colour = vec4(1.0, 1.0, 1.0, 1.0);
uniform vec4 tiling_offset = vec4(1.0, 1.0, 0.0, 0.0);
uniform mat4 mvp = mat4(1.0);

out vec3 vertex_colour;
out vec3 normal;
out vec2 uv;
out vec4 frag_colour;

void main() {
    gl_Position = vec4(in_position, 1.0) * mvp;
    vertex_colour = in_colour;
    frag_colour = colour;
    normal = in_normal;
    uv = in_uv * tiling_offset.xy + tiling_offset.zw;
}
