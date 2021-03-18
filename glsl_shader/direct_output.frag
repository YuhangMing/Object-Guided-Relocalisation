#version 330

in vec3 shaded_colour;
out vec4 colour_out;

void main()
{
    colour_out = vec4(shaded_colour, 1);
}