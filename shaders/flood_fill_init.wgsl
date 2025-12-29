#import "structs.wgsl"
#import "queue.wgsl"
#import "constants.wgsl"
#import "random.wgsl"
#import "helpers.wgsl"

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> q: Queue;
@group(0) @binding(2) var<storage, read_write> visited_buffer: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> work_count: atomic<i32>;
@group(0) @binding(4) var<storage, read_write> seeds: array<vec2u>;

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

    // Initialize or perturb seeds and enqueue them
    for (var i = global_id.x; i < SEED_COUNT; i += stride) {
        randseed = wang_hash(i) ^ wang_hash(frame_number());

        var pos_f: vec2f;

        if (frame_number() == 0u) {
            pos_f = frand_unorm2() * resolution_f();
        } else {
            let prev_pos = vec2f(seeds[i]);
            let offset = round(frand_snorm2() * 3.0);
            pos_f = clamp(prev_pos + offset, vec2f(0.0), resolution_f() - 1.0);
        }

        seeds[i] = vec2u(pos_f);
        
        let idx = pixel_index_f(pos_f);
        
        atomicStore(&visited_buffer[idx], i);
        loop { if (enqueue(&q, idx)) { break; } }
    }
}
