#version 330
out vec4 frag_out;
in vec3 vertex_colour;
in vec4 frag_colour;
in vec2 uv;
uniform sampler2D maintex;

void main() {
    frag_out = vec4(vertex_colour, 1.0) * frag_colour;
    frag_out *= texture(maintex, uv);
}
