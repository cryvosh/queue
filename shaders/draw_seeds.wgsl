#import "structs.wgsl"
#import "helpers.wgsl"
#import "constants.wgsl"

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> seeds: array<Seed>;
@group(0) @binding(2) var<storage, read_write> output_buffer: array<vec4f>;

const wg_size = 128;
@compute @workgroup_size(wg_size)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let stride = num_workgroups.x * wg_size;

    for (var i = global_id.x; i < SEED_COUNT; i += stride) {
        let center = seeds[i].pos;
        let radius = 3.0;
        
        let min_p = vec2i(max(vec2f(0.0), center - radius));
        let max_p = vec2i(min(resolution_f() - 1.0, center + radius));

        for (var y = min_p.y; y <= max_p.y; y++) {
            for (var x = min_p.x; x <= max_p.x; x++) {
                let pixel_pos = vec2f(f32(x), f32(y));
                let dist = distance(pixel_pos, center);
                
                if (dist <= radius) {
                    let idx = pixel_index_i(vec2i(x, y));
                    
                    output_buffer[idx] = vec4f(1.0);
                }
            }
        }
    }
}
