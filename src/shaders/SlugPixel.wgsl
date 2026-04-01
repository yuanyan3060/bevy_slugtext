// ===================================================
// Reference pixel shader for the Slug algorithm ported to WGSL
// ===================================================

const kLogBandTextureWidth: u32 = 12u;

fn TexelLoad2D_f32(tex: texture_2d<f32>, coords: vec2<i32>) -> vec4<f32> {
    return textureLoad(tex, coords, 0);
}

fn TexelLoad2D_u32(tex: texture_2d<u32>, coords: vec2<i32>) -> vec4<u32> {
    return textureLoad(tex, coords, 0);
}

fn CalcRootCode(y1: f32, y2: f32, y3: f32) -> u32 {
    let i1 = bitcast<u32>(y1) >> 31u;
    let i2 = bitcast<u32>(y2) >> 30u;
    let i3 = bitcast<u32>(y3) >> 29u;
    let shift = (i3 & 4u) | (((i2 & 2u) | (i1 & ~2u)) & ~4u);
    return ((0x2E74u >> shift) & 0x0101u);
}

fn SolveHorizPoly(p12: vec4<f32>, p3: vec2<f32>) -> vec2<f32> {
    let a = vec2<f32>(p12.x - p12.z * 2.0 + p3.x, p12.y - p12.w * 2.0 + p3.y);
    let b = vec2<f32>(p12.x - p12.z, p12.y - p12.w);
    let ra = 1.0 / a.y;
    let rb = 0.5 / b.y;
    let d = sqrt(max(b.y * b.y - a.y * p12.y, 0.0));
    var t1 = (b.y - d) * ra;
    var t2 = (b.y + d) * ra;
    if (abs(a.y) < 1.0 / 65536.0) {
        t1 = p12.y * rb;
        t2 = p12.y * rb;
    }
    return vec2<f32>((a.x * t1 - b.x * 2.0) * t1 + p12.x, (a.x * t2 - b.x * 2.0) * t2 + p12.x);
}

fn SolveVertPoly(p12: vec4<f32>, p3: vec2<f32>) -> vec2<f32> {
    let a = vec2<f32>(p12.x - p12.z * 2.0 + p3.x, p12.y - p12.w * 2.0 + p3.y);
    let b = vec2<f32>(p12.x - p12.z, p12.y - p12.w);
    let ra = 1.0 / a.x;
    let rb = 0.5 / b.x;
    let d = sqrt(max(b.x * b.x - a.x * p12.x, 0.0));
    var t1 = (b.x - d) * ra;
    var t2 = (b.x + d) * ra;
    if (abs(a.x) < 1.0 / 65536.0) {
        t1 = p12.x * rb;
        t2 = p12.x * rb;
    }
    return vec2<f32>((a.y * t1 - b.y * 2.0) * t1 + p12.y, (a.y * t2 - b.y * 2.0) * t2 + p12.y);
}

fn CalcBandLoc(glyphLoc: vec2<i32>, offset: u32) -> vec2<i32> {
    var bandLoc = vec2<i32>(glyphLoc.x + i32(offset), glyphLoc.y);
    bandLoc.y += bandLoc.x >> kLogBandTextureWidth;
    bandLoc.x &= (1 << kLogBandTextureWidth) - 1;
    return bandLoc;
}

@group(#{MATERIAL_BIND_GROUP}) @binding(100) var curveTexture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(101) var bandTexture: texture_2d<u32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(102) var<uniform> color: vec4<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(103) var<uniform> bg_color: vec4<f32>;

fn SlugRender(curveData: texture_2d<f32>, bandData: texture_2d<u32>, renderCoord: vec2<f32>, bandTransform: vec4<f32>, glyphData: vec4<i32>) -> f32 {
    let emsPerPixel = fwidth(renderCoord);
    let pixelsPerEm = 1.0 / emsPerPixel;
    let bandMax = vec2<i32>(glyphData.z, glyphData.w & 0x00FF);
    let bandIndex = clamp(vec2<i32>(renderCoord * bandTransform.xy + bandTransform.zw), vec2<i32>(0, 0), bandMax);
    let glyphLoc = vec2<i32>(glyphData.x, glyphData.y);

    var xcov: f32 = 0.0;
    var xwgt: f32 = 0.0;
    let hbandData = TexelLoad2D_u32(bandData, vec2<i32>(glyphLoc.x + bandIndex.y, glyphLoc.y)).xy;
    let hbandLoc = CalcBandLoc(glyphLoc, hbandData.y);
    for (var curveIndex = 0; curveIndex < i32(hbandData.x); curveIndex++) {
        let curveLoc = vec2<i32>(TexelLoad2D_u32(bandData, vec2<i32>(hbandLoc.x + curveIndex, hbandLoc.y)).xy);
        let p12 = TexelLoad2D_f32(curveData, curveLoc) - vec4<f32>(renderCoord, renderCoord);
        let p3 = TexelLoad2D_f32(curveData, vec2<i32>(curveLoc.x + 1, curveLoc.y)).xy - renderCoord;
        if (max(max(p12.x, p12.z), p3.x) * pixelsPerEm.x < -0.5) {
            break;
        }
        let code = CalcRootCode(p12.y, p12.w, p3.y);
        if (code != 0u) {
            let r = SolveHorizPoly(p12, p3) * pixelsPerEm.x;
            if ((code & 1u) != 0u) {
                xcov += saturate(r.x + 0.5);
                xwgt = max(xwgt, saturate(1.0 - abs(r.x) * 2.0));
            }
            if (code > 1u) {
                xcov -= saturate(r.y + 0.5);
                xwgt = max(xwgt, saturate(1.0 - abs(r.y) * 2.0));
            }
        }
    }

    var ycov: f32 = 0.0;
    var ywgt: f32 = 0.0;
    let vbandData = TexelLoad2D_u32(bandData, vec2<i32>(glyphLoc.x + bandMax.y + 1 + bandIndex.x, glyphLoc.y)).xy;
    let vbandLoc = CalcBandLoc(glyphLoc, vbandData.y);
    for (var curveIndex = 0; curveIndex < i32(vbandData.x); curveIndex++) {
        let curveLoc = vec2<i32>(TexelLoad2D_u32(bandData, vec2<i32>(vbandLoc.x + curveIndex, vbandLoc.y)).xy);
        let p12 = TexelLoad2D_f32(curveData, curveLoc) - vec4<f32>(renderCoord, renderCoord);
        let p3 = TexelLoad2D_f32(curveData, vec2<i32>(curveLoc.x + 1, curveLoc.y)).xy - renderCoord;
        if (max(max(p12.y, p12.w), p3.y) * pixelsPerEm.y < -0.5) {
            break;
        }
        let code = CalcRootCode(p12.x, p12.z, p3.x);
        if (code != 0u) {
            let r = SolveVertPoly(p12, p3) * pixelsPerEm.y;
            if ((code & 1u) != 0u) {
                ycov -= saturate(r.x + 0.5);
                ywgt = max(ywgt, saturate(1.0 - abs(r.x) * 2.0));
            }
            if (code > 1u) {
                ycov += saturate(r.y + 0.5);
                ywgt = max(ywgt, saturate(1.0 - abs(r.y) * 2.0));
            }
        }
    }

    let coverage = max(abs(xcov * xwgt + ycov * ywgt) / max(xwgt + ywgt, 1.0 / 65536.0), min(abs(xcov), abs(ycov)));
    return coverage;
}

struct VertexStruct {
    @builtin(position) position: vec4<f32>,
    @location(2) texcoord: vec2<f32>,
    @location(3) @interpolate(flat) banding: vec4<f32>,
    @location(4) @interpolate(flat) glyph: vec4<i32>,
};

@fragment
fn main(vresult: VertexStruct) -> @location(0) vec4<f32> {
    let coverage = SlugRender(curveTexture, bandTexture, vresult.texcoord, vresult.banding, vresult.glyph);
    return mix(bg_color, color, coverage);
}
