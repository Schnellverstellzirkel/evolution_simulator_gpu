struct Node { pos: vec2f, vel: vec2f, radius: f32, friction: f32, mass: f32, failed: f32 }
struct Muscle { a:u32, b:u32, short:f32, long:f32, period:f32, phase:f32, duty:f32, stiffness:f32 }
struct Meta { nodes:u32 }
struct NodeAdj { start:u32, count:u32 }
struct Params { tick:u32, steps:u32, stride:u32, count:u32, gravity:f32, air:f32, friction:f32, ground:f32, total_steps:u32, pad0:u32, pad1:u32, pad2:u32 }
struct Result {
    fitness:f32,
    ground_contact:f32,
    // Accumulate vertical min/max, then rewrite as range/cadence on the final tick.
    vertical_oscillation:f32,
    gait_frequency:f32,
    previous_center_y:f32,
    vertical_extremum:f32,
    vertical_trend:f32,
    gait_turns:f32,
}
@group(0) @binding(0) var<storage,read_write> nodes: array<Node>;
@group(0) @binding(1) var<storage,read> muscles: array<Muscle>;
@group(0) @binding(2) var<storage,read> metadata: array<Meta>;
@group(0) @binding(3) var<uniform> p: Params;
@group(0) @binding(4) var<storage,read_write> results: array<Result>;
@group(0) @binding(5) var<storage,read> node_adjacencies: array<NodeAdj>;
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
    return n;
}
@compute @workgroup_size(64)
fn advance(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3u) {
    let creature=(group.x*64u+lane)/p.stride;
    let local=lane%p.stride;let base=lane-local;
    var body=Meta(0u);
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
        if tick==200u {
            workgroupBarrier();
            positions[lane]=n.pos;velocities[lane]=n.vel;
            workgroupBarrier();
        }
        if local<body.nodes {
            let time=f32(max(tick,200u)-200u)/120.0;
            var force=vec2f(0.0);
            let adjacency=node_adjacencies[creature*p.stride+local];
            for(var j=0u;j<adjacency.count;j++) {
                let m=muscles[adjacency.start+j];
                let other=select(m.a,m.b,m.a==local);
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
        if creature<p.count && local==0u {
            if tick==0u {
            results[creature]=Result(0.0,0.0,1e20,-1e20,0.0,0.0,0.0,0.0);
            }
            if tick>=200u {
                var center_y=0.0;
                var contacts=0.0;
                for(var j=0u;j<body.nodes;j++) {
                    center_y+=positions[base+j].y;
                    if p.ground>0.0 && positions[base+j].y<=radii[base+j]+0.002 {contacts+=1.0;}
                }
                center_y/=f32(body.nodes);
                var r=results[creature];
                r.ground_contact+=contacts;
                r.vertical_oscillation=min(r.vertical_oscillation,center_y);
                r.gait_frequency=max(r.gait_frequency,center_y);
                if tick==200u {
                    r.previous_center_y=center_y;
                    r.vertical_extremum=center_y;
                    r.vertical_trend=0.0;
                    r.gait_turns=0.0;
                } else if (tick-200u)%4u==0u {
                    let delta=center_y-r.previous_center_y;
                    if r.vertical_trend==0.0 {
                        if abs(delta)>0.0005 {
                            r.vertical_trend=select(-1.0,1.0,delta>0.0);
                            r.vertical_extremum=center_y;
                        }
                    } else if r.vertical_trend>0.0 {
                        if center_y>r.vertical_extremum {
                            r.vertical_extremum=center_y;
                        } else if r.vertical_extremum-center_y>0.005 {
                            r.gait_turns+=1.0;
                            r.vertical_trend=-1.0;
                            r.vertical_extremum=center_y;
                        }
                    } else {
                        if center_y<r.vertical_extremum {
                            r.vertical_extremum=center_y;
                        } else if center_y-r.vertical_extremum>0.005 {
                            r.gait_turns+=1.0;
                            r.vertical_trend=1.0;
                            r.vertical_extremum=center_y;
                        }
                    }
                    r.previous_center_y=center_y;
                }
                results[creature]=r;
            }
            if tick+1u==p.total_steps {
                var score=0.0;var failed=0.0;
                for(var j=0u;j<body.nodes;j++){score+=positions[base+j].x;failed+=failures[base+j];}
                var r=results[creature];
                if failed>0.0 {r.fitness=-1e20;} else {r.fitness=score/f32(body.nodes);}
                if p.total_steps>200u {
                    r.vertical_oscillation=max(r.gait_frequency-r.vertical_oscillation,0.0);
                    r.gait_frequency=r.gait_turns*0.5/(f32(p.total_steps-200u)/120.0);
                } else {
                    r.vertical_oscillation=0.0;
                    r.gait_frequency=0.0;
                }
                results[creature]=r;
            }
        }
    }
    if creature<p.count && local<body.nodes {nodes[creature*p.stride+local]=n;}
}
