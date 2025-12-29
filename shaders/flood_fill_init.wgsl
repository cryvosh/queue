#import "structs.wgsl"
#import "queue.wgsl"
#import "constants.wgsl"
#import "random.wgsl"
#import "helpers.wgsl"

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> q: Queue;
@group(0) @binding(2) var<storage, read_write> visited_buffer: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> work_count: atomic<i32>;
@group(0) @binding(4) var<storage, read_write> seeds: array<Seed>;

const wg_size: u32 = 128u;

@compute @workgroup_size(wg_size)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let stride = num_workgroups.x * wg_size;

    if (global_id.x == 0u) {
        atomicStore(&work_count, i32(SEED_COUNT));
    }

    for (var i = global_id.x; i < SEED_COUNT; i += stride) {
        randseed = wang_hash(i) ^ wang_hash(frame_number());

        var pos: vec2f;
        var vel: vec2f;

        if (frame_number() == 0u) {
            pos = frand_unorm2() * resolution_f();
            vel = frand_snorm2() * 0.5;
        } else {
            let seed = seeds[i];
            
            // integrate
            vel = seed.vel + frand_snorm2() * 0.01;
            pos = seed.pos + vel;
            
            // bounce off walls
            let max = resolution_f() - 1.0;
            if pos.x < 0.0 { pos.x = -pos.x; vel.x = -vel.x; }
            if pos.y < 0.0 { pos.y = -pos.y; vel.y = -vel.y; }
            if pos.x > max.x { pos.x = 2.0 * max.x - pos.x; vel.x = -vel.x; }
            if pos.y > max.y { pos.y = 2.0 * max.y - pos.y; vel.y = -vel.y; }
            
            pos = clamp(pos, vec2f(0.0), max);
        }

        // Store updated state
        seeds[i] = Seed(pos, vel);
        
        let idx = pixel_index_f(pos);
        
        atomicStore(&visited_buffer[idx], i);
        loop { if (enqueue(&q, idx)) { break; } }
    }
}
