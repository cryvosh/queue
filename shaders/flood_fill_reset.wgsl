#import "constants.wgsl"
#import "structs.wgsl"
#import "queue_meta.wgsl"
#import "helpers.wgsl"

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> queue: QueueNonatomic;
@group(0) @binding(2) var<storage, read_write> visited_buffer: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> work_count: atomic<i32>;

const wg_size: u32 = 256u;

@compute @workgroup_size(wg_size)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let stride = num_workgroups.x * wg_size;

    if frame_number() == 0u {
        if global_id.x == 0u {
            queue.head = 0u;
            queue.tail = 0u;
            queue.count = 0;
            atomicStore(&work_count, 0);
        }

        for (var i = global_id.x; i < QUEUE_CAPACITY; i += stride) {
            queue.ring[i] = QUEUE_UNUSED;
        }
    }

    for (var i = global_id.x; i < pixel_count(); i += stride) {
        atomicStore(&visited_buffer[i], UNVISITED);
    }
}
