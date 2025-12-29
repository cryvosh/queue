fn resolution_f() -> vec2f {
    return vec2f(f32(uniforms.resolution.x), f32(uniforms.resolution.y));
}

fn resolution_u() -> vec2u {
    return vec2u(uniforms.resolution.x, uniforms.resolution.y);
}

fn resolution_i() -> vec2i {
    return vec2i(i32(uniforms.resolution.x), i32(uniforms.resolution.y));
}

fn on_screen_f(pixel: vec2f) -> bool {
    return all(pixel >= vec2f(0.0)) && all(pixel < resolution_f());
}

fn on_screen_u(pixel: vec2u) -> bool {
    return all(pixel >= vec2u(0u)) && all(pixel < resolution_u());
}

fn on_screen_i(pixel: vec2i) -> bool {
    return all(pixel >= vec2i(0)) && all(pixel < resolution_i());
}

fn pixel_index_f(pixel: vec2f) -> u32 {
    return u32(pixel.y) * resolution_u().x + u32(pixel.x);
}

fn pixel_index_u(pixel: vec2u) -> u32 {
    return pixel.y * resolution_u().x + pixel.x;
}

fn pixel_index_i(pixel: vec2i) -> i32 {
    return pixel.y * resolution_i().x + pixel.x;
}

fn index_to_pixel_f(pixel_index: u32) -> vec2f {
    return vec2f(f32(pixel_index % resolution_u().x), f32(pixel_index / resolution_u().x));
}

fn index_to_pixel_u(pixel_index: u32) -> vec2u {
    return vec2u(pixel_index % resolution_u().x, pixel_index / resolution_u().x);
}

fn index_to_pixel_i(pixel_index: u32) -> vec2i {
    return vec2i(i32(pixel_index % resolution_u().x), i32(pixel_index / resolution_u().x));
}

fn pixel_count() -> u32 {
    return resolution_u().x * resolution_u().y;
}

fn frame_number() -> u32 {
    return uniforms.frame;
}

fn mymod(x: f32, y: f32) -> f32 {
    return x - y * floor(x / y);
}

fn mymodvec3f(x: vec3f, y: f32) -> vec3f {
    return vec3f(mymod(x.x, y), mymod(x.y, y), mymod(x.z, y));
}

fn hsl2rgb(c: vec3f) -> vec3f {
    var temp = mymodvec3f(c.x * 6.0 + vec3f(0.0, 4.0, 2.0), 6.0);
    var rgb = clamp(abs(temp - 3.0) - 1.0, vec3f(0.0), vec3f(1.0));
    return c.z + c.y * (rgb - 0.5) * (1.0 - abs(2.0 * c.z - 1.0));
}

// x between 0 and 1: blue (cold) -> red (hot)
fn heat_map(x: f32) -> vec3f {
    var h = (1.0 - clamp(x, 0.0, 1.0)) * 240.0;
    return hsl2rgb(vec3f(h / 360.0, 1.0, 0.5));
}
