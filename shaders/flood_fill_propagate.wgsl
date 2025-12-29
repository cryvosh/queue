#import "structs.wgsl"
#import "queue.wgsl"
#import "constants.wgsl"
#import "random.wgsl"
#import "helpers.wgsl"

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> queue: Queue;
@group(0) @binding(2) var<storage, read_write> visited_buffer: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> work_count: atomic<i32>;

const NEIGHBORS = array<vec2<i32>, 4>(
    vec2i(-1, 0), vec2i(1, 0), vec2i(0, -1), vec2i(0, 1),
);

var<workgroup> wg_dequeue_result: WgDequeueResult;
var<workgroup> wg_all_done: bool;
var<workgroup> wg_enqueue_total: atomic<u32>;
var<workgroup> wg_enqueue_base: u32;

fn unpack_distance(packed: u32) -> u32 { return packed >> 16u; }
fn unpack_seed_id(packed: u32) -> u32 { return packed & 0xFFFFu; }
fn pack_distance_and_seed(distance: u32, seed_id: u32) -> u32 {
    return (distance << 16u) | seed_id;
}

const wg_size = 32u;
@compute @workgroup_size(wg_size)
fn main(
    @builtin(local_invocation_index) local_index: u32) 
{
    var to_enqueue: array<u32, 4>;
    var enqueue_count: u32;
    var enqueue_offset: u32;
    
    loop {
        if local_index == 0u {
            wg_dequeue_result = wg_dequeue_reserve(&queue, wg_size);
            wg_all_done = wg_dequeue_result.count == 0u && atomicLoad(&work_count) <= 0;
            atomicStore(&wg_enqueue_total, 0u);
        }
        workgroupBarrier();
        
        if workgroupUniformLoad(&wg_all_done) { break; }
        
        enqueue_count = 0u;
        
        if (local_index < wg_dequeue_result.count) {
            let pixel_idx = wg_dequeue_consume(&queue, wg_dequeue_result.start, local_index);
            let current = atomicLoad(&visited_buffer[pixel_idx]);
            
            if (current == UNVISITED) { // relaxed atomics make this possible
                to_enqueue[0] = pixel_idx;
                enqueue_count = 1u;
            } else {
                let next_distance = unpack_distance(current) + 1u;
                let seed_id = unpack_seed_id(current);
                let next_value = pack_distance_and_seed(next_distance, seed_id);
                let pixel = index_to_pixel_i(pixel_idx);

                for (var i = 0u; i < 4u; i++) {
                    let neighbor_pixel = pixel + NEIGHBORS[i];
                    if (on_screen_i(neighbor_pixel)) {
                        let neighbor_idx = pixel_index_i(neighbor_pixel);
                        let old_value = atomicMin(&visited_buffer[neighbor_idx], next_value);
                        if (old_value > next_value) {
                            to_enqueue[enqueue_count] = u32(neighbor_idx);
                            enqueue_count++;
                        }
                    }
                }
                
                atomicAdd(&work_count, i32(enqueue_count) - 1);
            }
        }
        
        enqueue_offset = atomicAdd(&wg_enqueue_total, enqueue_count);
        workgroupBarrier();
        
        let total_to_enqueue = atomicLoad(&wg_enqueue_total);
        if local_index == 0u {
            wg_enqueue_base = wg_enqueue_reserve(&queue, total_to_enqueue);
        }
        workgroupBarrier();
        
        for (var i = 0u; i < enqueue_count; i++) {
            wg_enqueue_publish(&queue, wg_enqueue_base, enqueue_offset + i, to_enqueue[i]);
        }
        workgroupBarrier();
    }
}
