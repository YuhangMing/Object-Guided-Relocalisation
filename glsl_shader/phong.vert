#version 330 

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 a_normal;

uniform mat4 mvp_matrix;

out vec3 shaded_colour;

void main() 
{
	gl_Position =  mvp_matrix * vec4(position, 1.0);
	vec3 lightpos = vec3(5, 5, 5);
	const float ka = 0.3;
	const float kd = 0.5;
	const float ks = 0.2;
	const float n = 20.0;
	const float ax = 1.0;
	const float dx = 1.0;
	const float sx = 1.0;
	const float lx = 1.0;
	vec3 L = normalize(lightpos - position);
	vec3 V = normalize(vec3(0.0) - position);
	vec3 R = normalize(2 * a_normal * dot(a_normal, L) - L);
	float i1 = ax * ka * dx;
	float i2 = lx * kd * dx * max(0.0, dot(a_normal, L));
	float i3 = lx * ks * sx * pow(max(0.0, dot(R, V)), n);
	float Ix = max(0.0, min(255.0, i1 + i2 + i3));
	shaded_colour = vec3(Ix, Ix, Ix);
} 