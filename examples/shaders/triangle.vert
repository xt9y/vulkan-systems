#version 450
layout(push_constant) uniform Rotation {float angle;} state;
layout(location=0) out vec3 color;
void main() {
    vec2 points[3]=vec2[3](vec2(-.7,-.55),vec2(.7,-.55),vec2(0,.75));
    vec3 colors[3]=vec3[3](vec3(.95,.3,.35),vec3(.25,.9,.65),vec3(.4,.55,1));
    vec2 p=points[gl_VertexIndex];
    float c=cos(state.angle),s=sin(state.angle);
    gl_Position=vec4(c*p.x-s*p.y,s*p.x+c*p.y,0,1);
    color=colors[gl_VertexIndex];
}
