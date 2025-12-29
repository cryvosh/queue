// https://nvpro-samples.github.io/vk_mini_path_tracer/
var<private> randseed: u32 = 123u;
fn frand_snorm() -> f32 {
    randseed = randseed * 747796405u + 1u;
    var word = ((randseed >> ((randseed >> 28u) + 4u)) ^ randseed) * 277803737u;
    word = (word >> 22u) ^ word;
    return (f32(word) / 4294967295.0) * 2.0 - 1.0; // -1 to 1
}
fn frand_unorm() -> f32 { // 0 to 1
    return frand_snorm() * 0.5 + 0.5;
}
fn frandAB(a: f32, b: f32) -> f32 {
    return a + (b - a) * frand_unorm();
}

fn frand_snorm2() -> vec2f {
    return vec2f(frand_snorm(), frand_snorm());
}
fn frand_unorm2() -> vec2f {
    return vec2f(frand_unorm(), frand_unorm());
}

fn frand_snorm3() -> vec3f {
    return vec3f(frand_snorm(), frand_snorm(), frand_snorm());
}
fn frand_unorm3() -> vec3f {
    return vec3f(frand_unorm(), frand_unorm(), frand_unorm());
}

// High-quality hash function for seed generation
fn wang_hash(seed: u32) -> u32 {
    var value = seed;
    value = (value ^ 61u) ^ (value >> 16u);
    value *= 9u;
    value = value ^ (value >> 4u);
    value *= 0x27d4eb2du;
    value = value ^ (value >> 15u);
    return value;
}

