#import "structs.wgsl"
#import "queue.wgsl"
#import "constants.wgsl"
#import "random.wgsl"
#import "helpers.wgsl"

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> q: Queue;
@group(0) @binding(2) var<storage, read_write> visited_buffer: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> work_count: atomic<i32>;

const neighbors = array<vec2<i32>, 4>(
    vec2<i32>(-1, 0), vec2<i32>(1, 0), vec2<i32>(0, -1), vec2<i32>(0, 1),
);

const wg_size: u32 = 32u;

@compute @workgroup_size(wg_size)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    var batch_items: array<u32, 4>;
    
    loop {
        if (atomicLoad(&work_count) <= 0) { break; }

        var idx = 0u;
        
        if !dequeue(&q, &idx) {
            if (atomicLoad(&work_count) <= 0) { break; }
            continue;
        }

        // Read our current state (packed: dist << 16 | id)
        let my_packed = atomicLoad(&visited_buffer[idx]);
        
        if (my_packed == UNVISITED) {
            // Relaxed atomics make this a possibility. Handle be reenqueuing.
            loop { if (enqueue(&q, idx)) { break; } }
            continue;
        }

        let my_dist = my_packed >> 16u;
        let my_id = my_packed & 0xFFFFu;
        let next_dist = my_dist + 1u;
        let next_packed = (next_dist << 16u) | my_id;

        var batch_count = 0u;

        let pixel = index_to_pixel_i(idx);

        for (var n = 0u; n < 4u; n++) {
            let neighbor = pixel + neighbors[n];
            if !on_screen_i(neighbor) { continue; }

            let neighbor_idx = pixel_index_i(neighbor);

            let old_val = atomicMin(&visited_buffer[neighbor_idx], next_packed);            
            if (old_val > next_packed) {
                // Cell has been improved, enqueue it to propagate the change.
                batch_items[batch_count] = u32(neighbor_idx);
                batch_count++;
            }
        }

        if (batch_count > 0u) {
            loop { if (enqueue_batch(&q, batch_items, batch_count)) { break; } }
        }

        // We finished 1 item, spawned N items
        atomicAdd(&work_count, i32(batch_count) - 1);
    }
}
