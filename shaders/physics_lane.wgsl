struct Node { pos: vec2f, vel: vec2f, radius: f32, friction: f32, mass: f32, failed: f32 }
struct Muscle { a:u32, b:u32, short:f32, long:f32, inv_period:f32, phase:f32, duty:f32, stiffness:f32, inv_duty:f32, inv_complement:f32 }
struct Meta { nodes:u32 }
struct NodeAdj { start:u32, count:u32 }
struct Params { tick:u32, steps:u32, stride:u32, count:u32, gravity:f32, air:f32, friction:f32, ground:f32, total_steps:u32, groups_x:u32, pad1:u32, pad2:u32 }
struct Result {
    fitness:f32,
    ground_contact:f32,
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

fn muscle_length(m:Muscle,time:f32)->f32 {
    let phase=fract(time*m.inv_period+m.phase);
    var wave:f32;
    if phase<m.duty {wave=0.5+0.5*cos(3.14159265359*phase*m.inv_duty);}
    else {wave=0.5-0.5*cos(3.14159265359*(phase-m.duty)*m.inv_complement);}
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

// One creature per lane. Every node stays in private registers for the whole
// dispatch, so the step loop needs no workgroup barriers and no shuffles.
@compute @workgroup_size(64)
fn advance(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3u) {
    let creature=(group.y*p.groups_x+group.x)*64u+lane;
    if creature>=p.count {return;}
    let body=metadata[creature];
    let body_nodes=body.nodes;
    let base=creature*p.stride;
    var state:array<Node,MAXN>;
    var forces:array<vec2f,MAXN>;
    var adjacency:array<NodeAdj,MAXN>;
    for(var i=0u;i<MAXN;i++) {
        if i<body_nodes {state[i]=nodes[base+i];}
        else {state[i]=Node(vec2f(0.0),vec2f(0.0),0.0,0.0,1.0,0.0);}
        forces[i]=vec2f(0.0);
        adjacency[i]=node_adjacencies[base+i];
    }
    var metrics=Result(0.0,0.0,1e20,-1e20,0.0,0.0,0.0,0.0);
    if p.tick>0u {metrics=results[creature];}
    for(var s=0u;s<p.steps;s++) {
        let tick=p.tick+s;
        if tick==200u {
            var avg=0.0;var low=1e20;
            for(var i=0u;i<MAXN;i++) {
                if i<body_nodes {avg+=state[i].pos.x;low=min(low,state[i].pos.y-state[i].radius);}
            }
            let shift=vec2f(avg/f32(body_nodes),low);
            for(var i=0u;i<MAXN;i++) {
                if i<body_nodes {state[i].pos-=shift;state[i].vel=vec2f(0.0);}
            }
        }
        let time=f32(max(tick,200u)-200u)/120.0;
        for(var i=0u;i<MAXN;i++) {
            if i<body_nodes && state[i].failed<0.5 {
                var force=vec2f(0.0);
                let adjacency_i=adjacency[i];
                for(var j=0u;j<adjacency_i.count;j++) {
                    let m=muscles[adjacency_i.start+j];
                    let other=select(m.a,m.b,m.a==i);
                    let d=state[other].pos-state[i].pos;
                    let distance=max(length(d),1e-6);
                    let dir=d/distance;
                    let relative=dot(state[other].vel-state[i].vel,dir);
                    let f=clamp(clamp(distance-muscle_length(m,time),-0.25,0.25)*m.stiffness+relative*0.15,-30.0,30.0);
                    force+=dir*f;
                }
                forces[i]=force;
            }
        }
        var gravity=0.0;if tick>=200u {gravity=p.gravity;}
        for(var i=0u;i<MAXN;i++) {
            if i<body_nodes && state[i].failed<0.5 {
                var n=state[i];
                n.vel=(n.vel+(forces[i]/n.mass-vec2f(0.0,gravity))/120.0)*p.air;
                n.pos+=n.vel/120.0;
                if tick>=200u {n=collide(n);}
                if !all(abs(n.pos)<vec2f(1e6)) || !all(abs(n.vel)<vec2f(1e6)) {n.failed=1.0;n.pos=vec2f(0.0);n.vel=vec2f(0.0);}
                state[i]=n;
            }
        }
        if tick>=200u {
            var center_y=0.0;
            var contacts=0.0;
            for(var i=0u;i<MAXN;i++) {
                if i<body_nodes {
                    center_y+=state[i].pos.y;
                    if p.ground>0.0 && state[i].pos.y<=state[i].radius+0.002 {contacts+=1.0;}
                }
            }
            center_y/=f32(body_nodes);
            metrics.ground_contact+=contacts;
            metrics.vertical_oscillation=min(metrics.vertical_oscillation,center_y);
            metrics.gait_frequency=max(metrics.gait_frequency,center_y);
            if tick==200u {
                metrics.previous_center_y=center_y;
                metrics.vertical_extremum=center_y;
                metrics.vertical_trend=0.0;
                metrics.gait_turns=0.0;
            } else if (tick-200u)%4u==0u {
                let delta=center_y-metrics.previous_center_y;
                if metrics.vertical_trend==0.0 {
                    if abs(delta)>0.0005 {
                        metrics.vertical_trend=select(-1.0,1.0,delta>0.0);
                        metrics.vertical_extremum=center_y;
                    }
                } else if metrics.vertical_trend>0.0 {
                    if center_y>metrics.vertical_extremum {
                        metrics.vertical_extremum=center_y;
                    } else if metrics.vertical_extremum-center_y>0.005 {
                        metrics.gait_turns+=1.0;
                        metrics.vertical_trend=-1.0;
                        metrics.vertical_extremum=center_y;
                    }
                } else {
                    if center_y<metrics.vertical_extremum {
                        metrics.vertical_extremum=center_y;
                    } else if center_y-metrics.vertical_extremum>0.005 {
                        metrics.gait_turns+=1.0;
                        metrics.vertical_trend=1.0;
                        metrics.vertical_extremum=center_y;
                    }
                }
                metrics.previous_center_y=center_y;
            }
        }
        if tick+1u==p.total_steps {
            var score=0.0;var failed=0.0;
            for(var i=0u;i<MAXN;i++) {
                if i<body_nodes {score+=state[i].pos.x;failed+=state[i].failed;}
            }
            if failed>0.0 {metrics.fitness=-1e20;} else {metrics.fitness=score/f32(body_nodes);}
            if p.total_steps>200u {
                metrics.vertical_oscillation=max(metrics.gait_frequency-metrics.vertical_oscillation,0.0);
                metrics.gait_frequency=metrics.gait_turns*0.5/(f32(p.total_steps-200u)/120.0);
            } else {
                metrics.vertical_oscillation=0.0;
                metrics.gait_frequency=0.0;
            }
        }
    }
    results[creature]=metrics;
    for(var i=0u;i<MAXN;i++) {
        if i<body_nodes {nodes[base+i]=state[i];}
    }
}
