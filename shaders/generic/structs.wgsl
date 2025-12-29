#import "constants.wgsl"

struct Uniforms {
    time: f32,
    frame: u32,
    resolution: vec2<u32>,
    pad: vec4<u32>,
}