struct Node { pos: vec2f, vel: vec2f, radius: f32, friction: f32, mass: f32, failed: f32 }
struct Muscle { a:u32, b:u32, short:f32, long:f32, inv_period:f32, phase:f32, duty:f32, stiffness:f32, inv_duty:f32, inv_complement:f32 }
struct Meta { nodes:u32 }
struct NodeAdj { start:u32, count:u32 }
struct Params { tick:u32, steps:u32, stride:u32, count:u32, gravity:f32, air:f32, friction:f32, ground:f32, total_steps:u32, pad0:u32, pad1:u32, pad2:u32 }
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

@compute @workgroup_size(128)
fn advance(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3u) {
    let creature=group.x*128u+lane;
    if creature>=p.count {return;}
    let n_nodes=metadata[creature].nodes;
    let base=creature*p.stride;
    var pos:array<vec2f,MAXNODES>;
    var vel:array<vec2f,MAXNODES>;
    var rad:array<f32,MAXNODES>;
    var fric:array<f32,MAXNODES>;
    var mass:array<f32,MAXNODES>;
    var failed:array<f32,MAXNODES>;
    var adjacency:array<NodeAdj,MAXNODES>;
    for(var i=0u;i<n_nodes;i++) {
        let n=nodes[base+i];
        pos[i]=n.pos;vel[i]=n.vel;rad[i]=n.radius;fric[i]=n.friction;mass[i]=n.mass;failed[i]=n.failed;
        adjacency[i]=node_adjacencies[base+i];
    }
    var metrics=Result(0.0,0.0,1e20,-1e20,0.0,0.0,0.0,0.0);
    if p.tick>0u {metrics=results[creature];}
    for(var s=0u;s<p.steps;s++) {
        let tick=p.tick+s;
        if tick==200u {
            var avg=0.0;var low=1e20;
            for(var i=0u;i<n_nodes;i++) {
                avg+=pos[i].x;low=min(low,pos[i].y-rad[i]);
            }
            let shift=vec2f(avg/f32(n_nodes),low);
            for(var i=0u;i<n_nodes;i++) {
                pos[i]-=shift;vel[i]=vec2f(0.0);
            }
        }
        let time=f32(max(tick,200u)-200u)/120.0;
        var forces:array<vec2f,MAXNODES>;
        for(var i=0u;i<n_nodes;i++) {
            if failed[i]>=0.5 {
                forces[i]=vec2f(0.0);
                continue;
            }
            var force=vec2f(0.0);
            let adjacency_i=adjacency[i];
            for(var j=0u;j<adjacency_i.count;j++) {
                let m=muscles[adjacency_i.start+j];
                let other=select(m.a,m.b,m.a==i);
                let d=pos[other]-pos[i];let distance=max(length(d),1e-6);let dir=d/distance;
                let relative=dot(vel[other]-vel[i],dir);
                let f=clamp(clamp(distance-muscle_length(m,time),-0.25,0.25)*m.stiffness+relative*0.15,-30.0,30.0);
                force+=dir*f;
            }
            forces[i]=force;
        }
        var gravity=0.0;if tick>=200u {gravity=p.gravity;}
        for(var i=0u;i<n_nodes;i++) {
            if failed[i]>=0.5 {continue;}
            vel[i]=(vel[i]+(forces[i]/mass[i]-vec2f(0.0,gravity))/120.0)*p.air;
            pos[i]+=vel[i]/120.0;
            if tick>=200u {
                var node=Node(pos[i],vel[i],rad[i],fric[i],mass[i],failed[i]);
                node=collide(node);
                pos[i]=node.pos;vel[i]=node.vel;
            }
            if !all(abs(pos[i])<vec2f(1e6)) || !all(abs(vel[i])<vec2f(1e6)) {
                failed[i]=1.0;pos[i]=vec2f(0.0);vel[i]=vec2f(0.0);
            }
        }
        if tick>=200u {
            var center_y=0.0;
            var contacts=0.0;
            for(var i=0u;i<n_nodes;i++) {
                center_y+=pos[i].y;
                if p.ground>0.0 && pos[i].y<=rad[i]+0.002 {contacts+=1.0;}
            }
            center_y/=f32(n_nodes);
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
            var score=0.0;var any_failed=0.0;
            for(var i=0u;i<n_nodes;i++) {
                score+=pos[i].x;any_failed+=failed[i];
            }
            if any_failed>0.0 {metrics.fitness=-1e20;} else {metrics.fitness=score/f32(n_nodes);}
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
    for(var i=0u;i<n_nodes;i++) {
        nodes[base+i]=Node(pos[i],vel[i],rad[i],fric[i],mass[i],failed[i]);
    }
}
