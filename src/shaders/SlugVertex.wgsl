#import bevy_pbr::mesh_view_bindings::view
#import bevy_pbr::mesh_bindings::mesh
#import bevy_pbr::mesh_functions
#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_clip}


// ===================================================
// Reference vertex shader for the Slug algorithm ported to WGSL
// ===================================================

struct SlugUnpackResult {
    vbnd: vec4<f32>,
    vgly: vec4<i32>,
}

fn SlugUnpack(tex: vec4<f32>, bnd: vec4<f32>) -> SlugUnpackResult {
    let g = vec2<u32>(bitcast<u32>(tex.z), bitcast<u32>(tex.w));
    let vgly = vec4<i32>(
        i32(g.x & 0xFFFFu),
        i32(g.x >> 16u),
        i32(g.y & 0xFFFFu),
        i32(g.y >> 16u)
    );
    return SlugUnpackResult(bnd, vgly);
}

struct SlugDilateResult {
    texcoord: vec2<f32>,
    vpos: vec2<f32>,
}

fn SlugDilate(pos: vec4<f32>, tex: vec4<f32>, jac: vec4<f32>, m0: vec4<f32>, m1: vec4<f32>, m3: vec4<f32>, dim: vec2<f32>) -> SlugDilateResult {
    let n = normalize(pos.zw);
    let s = dot(m3.xy, pos.xy) + m3.w;
    let t = dot(m3.xy, n);

    let u = (s * dot(m0.xy, n) - t * (dot(m0.xy, pos.xy) + m0.w)) * dim.x;
    let v = (s * dot(m1.xy, n) - t * (dot(m1.xy, pos.xy) + m1.w)) * dim.y;

    let s2 = s * s;
    let st = s * t;
    let uv = u * u + v * v;
    let d = pos.zw * (s2 * (st + sqrt(uv)) / (uv - st * st));

    let vpos = pos.xy + d;
    let texcoord = vec2<f32>(tex.x + dot(d, jac.xy), tex.y + dot(d, jac.zw));
    return SlugDilateResult(texcoord, vpos);
}

struct VertexInput {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(8) pos: vec4<f32>,
    @location(9) tex: vec4<f32>,
    @location(10) jac: vec4<f32>,
    @location(11) bnd: vec4<f32>,
};

struct VertexStruct {
    @builtin(position) position: vec4<f32>,
    @location(2) texcoord: vec2<f32>,
    @location(3) @interpolate(flat) banding: vec4<f32>,
    @location(4) @interpolate(flat) glyph: vec4<i32>,
};

@vertex
fn main(attrib: VertexInput) -> VertexStruct {
    var vresult: VertexStruct;

    let slug_viewport = view.viewport.zw;
    let model_matrix = mesh_functions::get_world_from_local(attrib.instance_index);
    
    let sm = transpose(view.clip_from_world * model_matrix);

    let m0 = sm[0];
    let m1 = sm[1];
    let m2 = sm[2];
    let m3 = sm[3];
    
    let dilateResult = SlugDilate(attrib.pos, attrib.tex, attrib.jac, m0, m1, m3, slug_viewport);
    vresult.texcoord = dilateResult.texcoord;
    let p = dilateResult.vpos;

    vresult.position = vec4<f32>(
        p.x * m0.x + p.y * m0.y + m0.w,
        p.x * m1.x + p.y * m1.y + m1.w,
        p.x * m2.x + p.y * m2.y + m2.w,
        p.x * m3.x + p.y * m3.y + m3.w
    );

    let unpackResult = SlugUnpack(attrib.tex, attrib.bnd);
    vresult.banding = unpackResult.vbnd;
    vresult.glyph = unpackResult.vgly;
    return vresult;
}