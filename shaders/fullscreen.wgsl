#import "structs.wgsl"
#import "helpers.wgsl"
#import "constants.wgsl"

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> visited_buffer: array<u32>;
@group(0) @binding(2) var<storage, read_write> output_buffer: array<vec4f>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let pixel = vec2u(gid.x, gid.y);
    
    if !on_screen_u(pixel) {
        return;
    }
    
    let idx = pixel_index_u(pixel);
    
    let packed_val = visited_buffer[idx];    
    let id = packed_val & 0xFFFFu;
    
    var color = vec3f(heat_map(f32(id)/f32(SEED_COUNT)));
    
    output_buffer[idx] = vec4f(color, 1.0);
}
