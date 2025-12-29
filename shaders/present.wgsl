#import "structs.wgsl"
#import "helpers.wgsl"

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> output_buffer: array<vec4f>;

@vertex 
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32
) -> @builtin(position) vec4f {
    var pos = vec2f(f32((in_vertex_index << 1u) & 2u), f32(in_vertex_index & 2u)) * 2.0 - 1.0;
    return vec4f(pos.x, pos.y, 0.0, 1.0);
}

@fragment 
fn fs_main(
    @builtin(position) coord: vec4f
) -> @location(0) vec4f {
    let pixel_idx = pixel_index_f(coord.xy);
    let color = output_buffer[pixel_idx].xyz;
    return vec4f(color, 1.0);
}
