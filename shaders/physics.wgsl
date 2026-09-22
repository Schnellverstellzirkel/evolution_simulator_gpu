struct Node { pos: vec2f, vel: vec2f, radius: f32, friction: f32, mass: f32, failed: f32 }
struct Muscle { a:u32, b:u32, short:f32, long:f32, period:f32, phase:f32, duty:f32, stiffness:f32 }
struct Meta { nodes:u32, muscles:u32, start:u32, pad:u32 }
struct Params { tick:u32, steps:u32, stride:u32, count:u32, gravity:f32, air:f32, friction:f32, ground:f32, obstacles:u32, pad0:u32, pad1:u32, pad2:u32 }
@group(0) @binding(0) var<storage,read_write> nodes: array<Node>;
@group(0) @binding(1) var<storage,read> muscles: array<Muscle>;
@group(0) @binding(2) var<storage,read> metadata: array<Meta>;
@group(0) @binding(3) var<uniform> p: Params;
@group(0) @binding(4) var<storage,read> obstacles: array<vec4f>;
@group(0) @binding(5) var<storage,read_write> fitness: array<f32>;
var<workgroup> positions: array<vec2f,64>;
var<workgroup> velocities: array<vec2f,64>;
var<workgroup> radii: array<f32,64>;
var<workgroup> failures: array<f32,64>;

fn muscle_length(m:Muscle,time:f32)->f32 {
    let phase=fract(time/m.period+m.phase);
    var wave:f32;
    if phase<m.duty {wave=0.5+0.5*cos(3.14159265359*phase/m.duty);}
    else {wave=0.5-0.5*cos(3.14159265359*(phase-m.duty)/(1.0-m.duty));}
    return mix(m.short,m.long,wave);
}
fn contact(node:Node, normal:vec2f, penetration:f32)->Node {
    var n=node;n.pos+=normal*penetration;let vn=dot(n.vel,normal);
    if vn<0.0 {n.vel-=vn*normal;let keep=max(0.0,1.0-(-vn)*n.friction*p.friction/max(length(n.vel),1e-8));n.vel*=keep;}
    return n;
}
fn collide(node:Node)->Node {
    var n=node;
    if p.ground>0.0 && n.pos.y<n.radius {n=contact(n,vec2f(0.0,1.0),n.radius-n.pos.y);}
    for(var j=0u;j<p.obstacles;j++) {
        let r=obstacles[j];let q=clamp(n.pos,r.xy,r.zw);let d=n.pos-q;let distance=length(d);
        if distance>1e-8 && distance<n.radius {n=contact(n,d/distance,n.radius-distance);}
        else if distance<=1e-8 {
            let ds=vec4f(n.pos.x-r.x,r.z-n.pos.x,n.pos.y-r.y,r.w-n.pos.y);
            var side=0u;for(var k=1u;k<4u;k++){if ds[k]<ds[side]{side=k;}}
            let normals=array<vec2f,4>(vec2f(-1,0),vec2f(1,0),vec2f(0,-1),vec2f(0,1));
            n=contact(n,normals[side],n.radius+ds[side]);
        }
    }
    return n;
}
@compute @workgroup_size(64)
fn advance(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3u) {
    let creature=(group.x*64u+lane)/p.stride;
    let local=lane%p.stride;let base=lane-local;
    var body=Meta(0u,0u,0u,0u);
    var n=Node(vec2f(0.0),vec2f(0.0),0.0,0.0,1.0,0.0);
    if creature<p.count {body=metadata[creature];if local<body.nodes {n=nodes[creature*p.stride+local];}}
    positions[lane]=n.pos;velocities[lane]=n.vel;radii[lane]=n.radius;failures[lane]=n.failed;
    workgroupBarrier();
    for(var s=0u;s<p.steps;s++) {
        let tick=p.tick+s;
        if tick==200u {
            if local<body.nodes {
                var avg=0.0;var low=1e20;
                for(var j=0u;j<body.nodes;j++){avg+=positions[base+j].x;low=min(low,positions[base+j].y-radii[base+j]);}
                n.pos-=vec2f(avg/f32(body.nodes),low);n.vel=vec2f(0.0);
            }
        }
        workgroupBarrier();
        positions[lane]=n.pos;velocities[lane]=n.vel;
        workgroupBarrier();
        if local<body.nodes {
            let time=f32(max(tick,200u)-200u)/120.0;
            var force=vec2f(0.0);
            for(var j=0u;j<body.muscles;j++) {
                let m=muscles[body.start+j];var other:u32;
                if m.a==local {other=m.b;} else if m.b==local {other=m.a;} else {continue;}
                let d=positions[base+other]-n.pos;let distance=max(length(d),1e-6);let dir=d/distance;
                let relative=dot(velocities[base+other]-n.vel,dir);
                let f=clamp(clamp(distance-muscle_length(m,time),-0.25,0.25)*m.stiffness+relative*0.15,-30.0,30.0);
                force+=dir*f;
            }
            var gravity=0.0;if tick>=200u {gravity=p.gravity;}
            n.vel=(n.vel+(force/n.mass-vec2f(0.0,gravity))/120.0)*p.air;
            n.pos+=n.vel/120.0;
            if tick>=200u {n=collide(n);}
            if !all(abs(n.pos)<vec2f(1e6)) || !all(abs(n.vel)<vec2f(1e6)) {n.failed=1.0;n.pos=vec2f(0.0);n.vel=vec2f(0.0);}
        }
        workgroupBarrier();positions[lane]=n.pos;velocities[lane]=n.vel;failures[lane]=n.failed;workgroupBarrier();
    }
    if creature<p.count && local<body.nodes {nodes[creature*p.stride+local]=n;}
    if creature<p.count && local==0u {
        var score=0.0;var failed=0.0;
        for(var j=0u;j<body.nodes;j++){score+=positions[base+j].x;failed+=failures[base+j];}
        if failed>0.0 {fitness[creature]=-1e20;} else {fitness[creature]=score/f32(body.nodes);}
    }
}
