#version 330 

layout(location = 0) in vec3 position;
layout(location = 2) in vec3 a_colour;

uniform mat4 mvp_matrix;

out vec3 shaded_colour;

void main() 
{
	gl_Position = mvp_matrix * vec4(position, 1.0);
	shaded_colour = a_colour;
}