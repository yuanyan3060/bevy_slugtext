use std::cmp::Ordering;
use std::collections::HashMap;

use bevy::asset::RenderAssetUsages;
use bevy::image::Image;
use bevy::mesh::{Indices, Mesh, MeshVertexAttribute, PrimitiveTopology};
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use ttf_parser::{Face, GlyphId, OutlineBuilder, Rect};

const LINE_EPSILON: f32 = 0.125;
const TEX_WIDTH: usize = 4096;
const DEFAULT_BAND_COUNT: usize = 8;

const _: () = {
    assert!(Mesh::FIRST_AVAILABLE_CUSTOM_ATTRIBUTE == 8);
};

pub const ATTRIBUTE_SLUG_POS: MeshVertexAttribute = MeshVertexAttribute::new(
    "_SLUG_POS",
    Mesh::FIRST_AVAILABLE_CUSTOM_ATTRIBUTE,
    bevy::render::render_resource::VertexFormat::Float32x4,
);

pub const ATTRIBUTE_SLUG_TEX: MeshVertexAttribute = MeshVertexAttribute::new(
    "_SLUG_TEX",
    Mesh::FIRST_AVAILABLE_CUSTOM_ATTRIBUTE + 1,
    bevy::render::render_resource::VertexFormat::Float32x4,
);

pub const ATTRIBUTE_SLUG_JAC: MeshVertexAttribute = MeshVertexAttribute::new(
    "_SLUG_JAC",
    Mesh::FIRST_AVAILABLE_CUSTOM_ATTRIBUTE + 2,
    bevy::render::render_resource::VertexFormat::Float32x4,
);

pub const ATTRIBUTE_SLUG_BND: MeshVertexAttribute = MeshVertexAttribute::new(
    "_SLUG_BND",
    Mesh::FIRST_AVAILABLE_CUSTOM_ATTRIBUTE + 3,
    bevy::render::render_resource::VertexFormat::Float32x4,
);

#[derive(Clone, Copy, Debug)]

pub struct Bounds {
    pub x_min: f32,
    pub y_min: f32,
    pub x_max: f32,
    pub y_max: f32,
}

#[derive(Clone, Copy, Debug)]

pub struct QuadCurve {
    pub p0x: f32,
    pub p0y: f32,
    pub p1x: f32,
    pub p1y: f32,
    pub p2x: f32,
    pub p2y: f32,
}

#[derive(Clone, Debug, Default)]

pub struct BandEntry {
    pub curve_indices: Vec<usize>,
}

#[derive(Clone, Debug)]

pub struct GlyphBands {
    pub h_bands: Vec<BandEntry>,
    pub v_bands: Vec<BandEntry>,
    pub h_band_count: usize,
    pub v_band_count: usize,
}

#[derive(Clone, Debug)]

pub struct SlugGlyph {
    pub glyph_id: u16,
    pub curves: Vec<QuadCurve>,
    pub bands: GlyphBands,
    pub bounds: Bounds,
}

#[derive(Debug)]
pub struct PreparedText {
    pub slug_glyphs: Vec<SlugGlyph>,
    pub vertices: Vec<f32>,
    pub indices: Vec<u32>,
    pub curve_tex_data: Vec<f32>,
    pub band_tex_data: Vec<u32>,
    pub curve_tex_height: usize,
    pub band_tex_height: usize,
    pub total_advance: f32,
}

impl PreparedText {
    pub fn mesh(&self) -> Mesh {
        let mut position = Vec::new();
        let mut normal = Vec::new();

        let mut pos = Vec::new();
        let mut tex = Vec::new();
        let mut jac = Vec::new();
        let mut bnd = Vec::new();
        for chunk in self.vertices.chunks_exact(16) {
            position.push([chunk[0], chunk[1], 0.0]);
            normal.push([chunk[2], chunk[3], 0.0]);

            pos.push([chunk[0], chunk[1], chunk[2], chunk[3]]);
            tex.push([chunk[4], chunk[5], chunk[6], chunk[7]]);
            jac.push([chunk[8], chunk[9], chunk[10], chunk[11]]);
            bnd.push([chunk[12], chunk[13], chunk[14], chunk[15]]);
        }

        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, position)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normal)
        .with_inserted_attribute(ATTRIBUTE_SLUG_POS, pos)
        .with_inserted_attribute(ATTRIBUTE_SLUG_TEX, tex)
        .with_inserted_attribute(ATTRIBUTE_SLUG_JAC, jac)
        .with_inserted_attribute(ATTRIBUTE_SLUG_BND, bnd)
        .with_inserted_indices(Indices::U32(self.indices.clone()))
    }

    pub fn curve(&self) -> Image {
        let curve_extent = Extent3d {
            width: TEX_WIDTH as u32,
            height: self.curve_tex_height as u32,
            depth_or_array_layers: 1,
        };

        let mut data = Vec::with_capacity(self.curve_tex_data.len() * 4);
        for &i in &self.curve_tex_data {
            data.extend_from_slice(&i.to_le_bytes());
        }

        let mut curve_image = Image::new(
            curve_extent,
            TextureDimension::D2,
            data,
            TextureFormat::Rgba32Float,
            RenderAssetUsages::RENDER_WORLD,
        );
        curve_image.texture_descriptor.usage =
            TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
        curve_image
    }

    pub fn band(&self) -> Image {
        let band_extent = Extent3d {
            width: TEX_WIDTH as u32,
            height: self.band_tex_height as u32,
            depth_or_array_layers: 1,
        };

        let mut data = Vec::with_capacity(self.band_tex_data.len() * 4);
        for &i in &self.band_tex_data {
            data.extend_from_slice(&i.to_le_bytes());
        }

        let mut band_image = Image::new(
            band_extent,
            TextureDimension::D2,
            data,
            TextureFormat::Rgba32Uint,
            RenderAssetUsages::RENDER_WORLD,
        );
        band_image.texture_descriptor.usage =
            TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
        band_image
    }
}

pub struct GlyphPlacement {
    pub glyph_id: GlyphId,
    pub x_advance: f32,
    pub x_offset: f32,
    pub y_offset: f32,
}

#[derive(Clone, Copy)]
pub struct GlyphBandInfo {
    pub glyph_loc_x: u16,
    pub glyph_loc_y: u16,
}

pub struct PackedGlyphData {
    pub curve_tex_data: Vec<f32>,
    pub band_tex_data: Vec<u32>,
    pub curve_tex_height: usize,
    pub band_tex_height: usize,
    pub glyph_band_info: Vec<GlyphBandInfo>,
    pub glyph_curve_starts: Vec<usize>,
}

#[derive(Default)]
pub struct CurveBuilder {
    pub curves: Vec<QuadCurve>,
    pub cur_x: f32,
    pub cur_y: f32,
    pub start_x: f32,
    pub start_y: f32,
}

impl CurveBuilder {
    fn into_curves(self) -> Vec<QuadCurve> {
        self.curves
    }
}

impl OutlineBuilder for CurveBuilder {
    fn move_to(&mut self, x: f32, y: f32) {
        self.cur_x = x;
        self.cur_y = y;
        self.start_x = x;
        self.start_y = y;
    }

    fn line_to(&mut self, x: f32, y: f32) {
        if let Some(curve) = line_to_quadratic(self.cur_x, self.cur_y, x, y) {
            self.curves.push(curve);
        }
        self.cur_x = x;
        self.cur_y = y;
    }

    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        self.curves.push(QuadCurve {
            p0x: self.cur_x,
            p0y: self.cur_y,
            p1x: x1,
            p1y: y1,
            p2x: x,
            p2y: y,
        });
        self.cur_x = x;
        self.cur_y = y;
    }

    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        let m01x = (self.cur_x + x1) * 0.5;
        let m01y = (self.cur_y + y1) * 0.5;
        let m12x = (x1 + x2) * 0.5;
        let m12y = (y1 + y2) * 0.5;
        let m23x = (x2 + x) * 0.5;
        let m23y = (y2 + y) * 0.5;
        let m012x = (m01x + m12x) * 0.5;
        let m012y = (m01y + m12y) * 0.5;
        let m123x = (m12x + m23x) * 0.5;
        let m123y = (m12y + m23y) * 0.5;
        let midx = (m012x + m123x) * 0.5;
        let midy = (m012y + m123y) * 0.5;

        self.curves.push(QuadCurve {
            p0x: self.cur_x,
            p0y: self.cur_y,
            p1x: m01x,
            p1y: m01y,
            p2x: midx,
            p2y: midy,
        });
        self.curves.push(QuadCurve {
            p0x: midx,
            p0y: midy,
            p1x: m123x,
            p1y: m123y,
            p2x: x,
            p2y: y,
        });
        self.cur_x = x;
        self.cur_y = y;
    }

    fn close(&mut self) {
        let dx = self.start_x - self.cur_x;
        let dy = self.start_y - self.cur_y;
        if dx.abs() > 0.1 || dy.abs() > 0.1 {
            if let Some(curve) =
                line_to_quadratic(self.cur_x, self.cur_y, self.start_x, self.start_y)
            {
                self.curves.push(curve);
            }
        }
        self.cur_x = self.start_x;
        self.cur_y = self.start_y;
    }
}

fn line_to_quadratic(x0: f32, y0: f32, x1: f32, y1: f32) -> Option<QuadCurve> {
    let dx = x1 - x0;
    let dy = y1 - y0;
    if dx.abs() < 0.1 && dy.abs() < 0.1 {
        return None;
    }

    let mx = (x0 + x1) * 0.5;
    let my = (y0 + y1) * 0.5;

    if dx.abs() > 0.1 && dy.abs() > 0.1 {
        let length = (dx * dx + dy * dy).sqrt();
        if length > 0.0 {
            let inv_length = LINE_EPSILON / length;
            return Some(QuadCurve {
                p0x: x0,
                p0y: y0,
                p1x: mx - dy * inv_length,
                p1y: my + dx * inv_length,
                p2x: x1,
                p2y: y1,
            });
        }
    }

    Some(QuadCurve {
        p0x: x0,
        p0y: y0,
        p1x: mx,
        p1y: my,
        p2x: x1,
        p2y: y1,
    })
}

fn rect_to_bounds(rect: Rect) -> Bounds {
    Bounds {
        x_min: rect.x_min as f32,
        y_min: rect.y_min as f32,
        x_max: rect.x_max as f32,
        y_max: rect.y_max as f32,
    }
}

fn extract_curves(face: &Face<'_>, glyph_id: GlyphId) -> Option<(Vec<QuadCurve>, Bounds)> {
    let rect = face.glyph_bounding_box(glyph_id)?;
    let mut builder = CurveBuilder::default();
    face.outline_glyph(glyph_id, &mut builder)?;
    Some((builder.into_curves(), rect_to_bounds(rect)))
}

fn build_bands(curves: &[QuadCurve], bounds: &Bounds, band_count: usize) -> GlyphBands {
    let mut h_bands = vec![BandEntry::default(); band_count];
    let mut v_bands = vec![BandEntry::default(); band_count];

    let width = bounds.x_max - bounds.x_min;
    let height = bounds.y_max - bounds.y_min;
    let h_inv = if height.abs() > f32::EPSILON {
        1.0 / height
    } else {
        0.0
    };
    let w_inv = if width.abs() > f32::EPSILON {
        1.0 / width
    } else {
        0.0
    };

    for (ci, curve) in curves.iter().enumerate() {
        let cy_min = curve.p0y.min(curve.p1y).min(curve.p2y);
        let cy_max = curve.p0y.max(curve.p1y).max(curve.p2y);
        let cx_min = curve.p0x.min(curve.p1x).min(curve.p2x);
        let cx_max = curve.p0x.max(curve.p1x).max(curve.p2x);

        if height > 0.0 {
            let norm_min = (cy_min - bounds.y_min) * h_inv;
            let norm_max = (cy_max - bounds.y_min) * h_inv;
            let mut b0 = (norm_min * band_count as f32).floor() as isize;
            let mut b1 = (norm_max * band_count as f32).floor() as isize;
            b0 = b0.clamp(0, (band_count - 1) as isize);
            b1 = b1.clamp(0, (band_count - 1) as isize);
            for b in b0..=b1 {
                h_bands[b as usize].curve_indices.push(ci);
            }
        }

        if width > 0.0 {
            let norm_min = (cx_min - bounds.x_min) * w_inv;
            let norm_max = (cx_max - bounds.x_min) * w_inv;
            let mut b0 = (norm_min * band_count as f32).floor() as isize;
            let mut b1 = (norm_max * band_count as f32).floor() as isize;
            b0 = b0.clamp(0, (band_count - 1) as isize);
            b1 = b1.clamp(0, (band_count - 1) as isize);
            for b in b0..=b1 {
                v_bands[b as usize].curve_indices.push(ci);
            }
        }
    }

    GlyphBands {
        h_bands,
        v_bands,
        h_band_count: band_count,
        v_band_count: band_count,
    }
}

fn pack_glyph_data(glyphs: &[SlugGlyph]) -> PackedGlyphData {
    let total_curve_texels: usize = glyphs.iter().map(|g| g.curves.len() * 2).sum();
    let curve_texels = total_curve_texels.max(1);
    let curve_tex_height = ((curve_texels + TEX_WIDTH - 1) / TEX_WIDTH).max(1);
    let mut curve_tex_data = vec![0.0f32; TEX_WIDTH * curve_tex_height * 4];
    let mut curve_texel_idx = 0usize;
    let mut glyph_curve_starts = Vec::with_capacity(glyphs.len());

    for glyph in glyphs {
        glyph_curve_starts.push(curve_texel_idx);
        for curve in &glyph.curves {
            let i0 = curve_texel_idx;
            let x0 = i0 % TEX_WIDTH;
            let y0 = i0 / TEX_WIDTH;
            let off0 = (y0 * TEX_WIDTH + x0) * 4;
            curve_tex_data[off0] = curve.p0x;
            curve_tex_data[off0 + 1] = curve.p0y;
            curve_tex_data[off0 + 2] = curve.p1x;
            curve_tex_data[off0 + 3] = curve.p1y;

            let i1 = curve_texel_idx + 1;
            let x1 = i1 % TEX_WIDTH;
            let y1 = i1 / TEX_WIDTH;
            let off1 = (y1 * TEX_WIDTH + x1) * 4;
            curve_tex_data[off1] = curve.p2x;
            curve_tex_data[off1 + 1] = curve.p2y;

            curve_texel_idx += 2;
        }
    }

    let mut total_band_texels = 0usize;
    for glyph in glyphs {
        let header_count = glyph.bands.h_band_count + glyph.bands.v_band_count;
        if header_count == 0 {
            continue;
        }
        let cur_x = total_band_texels % TEX_WIDTH;
        if cur_x + header_count > TEX_WIDTH {
            total_band_texels += TEX_WIDTH - cur_x;
        }
        total_band_texels += header_count;
        for band in glyph.bands.h_bands.iter().chain(glyph.bands.v_bands.iter()) {
            total_band_texels += band.curve_indices.len();
        }
    }

    let band_texels = total_band_texels.max(1);
    let band_tex_height = ((band_texels + TEX_WIDTH - 1) / TEX_WIDTH).max(1);
    let mut band_tex_data = vec![0u32; TEX_WIDTH * band_tex_height * 4];
    let mut band_texel_idx = 0usize;
    let mut glyph_band_info = Vec::with_capacity(glyphs.len());

    for (glyph_index, glyph) in glyphs.iter().enumerate() {
        let h_band_count = glyph.bands.h_band_count;
        let v_band_count = glyph.bands.v_band_count;
        let header_count = h_band_count + v_band_count;
        if header_count == 0 {
            glyph_band_info.push(GlyphBandInfo {
                glyph_loc_x: 0,
                glyph_loc_y: 0,
            });
            continue;
        }

        let cur_x = band_texel_idx % TEX_WIDTH;
        if cur_x + header_count > TEX_WIDTH {
            band_texel_idx += TEX_WIDTH - cur_x;
        }

        let glyph_start = band_texel_idx;
        let glyph_loc_x = (glyph_start % TEX_WIDTH) as u16;
        let glyph_loc_y = (glyph_start / TEX_WIDTH) as u16;
        glyph_band_info.push(GlyphBandInfo {
            glyph_loc_x,
            glyph_loc_y,
        });

        let glyph_curve_start = glyph_curve_starts[glyph_index];
        let mut sorted_h = glyph
            .bands
            .h_bands
            .iter()
            .map(|band| {
                let mut indices = band.curve_indices.clone();
                indices.sort_by(|&a, &b| {
                    let ca = &glyph.curves[a];
                    let cb = &glyph.curves[b];
                    let max_a = ca.p0x.max(ca.p1x).max(ca.p2x);
                    let max_b = cb.p0x.max(cb.p1x).max(cb.p2x);
                    max_b.partial_cmp(&max_a).unwrap_or(Ordering::Equal)
                });
                BandEntry {
                    curve_indices: indices,
                }
            })
            .collect::<Vec<_>>();
        let mut sorted_v = glyph
            .bands
            .v_bands
            .iter()
            .map(|band| {
                let mut indices = band.curve_indices.clone();
                indices.sort_by(|&a, &b| {
                    let ca = &glyph.curves[a];
                    let cb = &glyph.curves[b];
                    let max_a = ca.p0y.max(ca.p1y).max(ca.p2y);
                    let max_b = cb.p0y.max(cb.p1y).max(cb.p2y);
                    max_b.partial_cmp(&max_a).unwrap_or(Ordering::Equal)
                });
                BandEntry {
                    curve_indices: indices,
                }
            })
            .collect::<Vec<_>>();

        let mut all_bands = Vec::with_capacity(sorted_h.len() + sorted_v.len());
        all_bands.append(&mut sorted_h);
        all_bands.append(&mut sorted_v);

        let mut curve_list_offset = header_count;
        let mut band_offsets = Vec::with_capacity(all_bands.len());
        for band in &all_bands {
            band_offsets.push(curve_list_offset);
            curve_list_offset += band.curve_indices.len();
        }

        for (i, band) in all_bands.iter().enumerate() {
            let tl = glyph_start + i;
            let tx = tl % TEX_WIDTH;
            let ty = tl / TEX_WIDTH;
            let di = (ty * TEX_WIDTH + tx) * 4;
            band_tex_data[di] = band.curve_indices.len() as u32;
            band_tex_data[di + 1] = band_offsets[i] as u32;
        }

        for (band_index, band) in all_bands.iter().enumerate() {
            let list_start = glyph_start + band_offsets[band_index];
            for (offset, curve_index) in band.curve_indices.iter().enumerate() {
                let curve_texel = glyph_curve_start + curve_index * 2;
                let c_tex_x = (curve_texel % TEX_WIDTH) as u32;
                let c_tex_y = (curve_texel / TEX_WIDTH) as u32;

                let tl = list_start + offset;
                let tx = tl % TEX_WIDTH;
                let ty = tl / TEX_WIDTH;
                let di = (ty * TEX_WIDTH + tx) * 4;
                band_tex_data[di] = c_tex_x;
                band_tex_data[di + 1] = c_tex_y;
            }
        }

        band_texel_idx = glyph_start + curve_list_offset;
    }

    PackedGlyphData {
        curve_tex_data,
        band_tex_data,
        curve_tex_height,
        band_tex_height,
        glyph_band_info,
        glyph_curve_starts,
    }
}

fn pack_u32_as_f32(value: u32) -> f32 {
    f32::from_bits(value)
}

fn shape_simple(face: &Face<'_>, text: &str) -> Vec<GlyphPlacement> {
    let mut glyphs = Vec::new();
    for ch in text.chars() {
        if let Some(glyph_id) = face.glyph_index(ch) {
            let advance = face
                .glyph_hor_advance(glyph_id)
                .map(|v| v as f32)
                .unwrap_or_else(|| face.units_per_em() as f32);
            glyphs.push(GlyphPlacement {
                glyph_id,
                x_advance: advance,
                x_offset: 0.0,
                y_offset: 0.0,
            });
        }
    }
    glyphs
}

pub fn prepare_text(face: &Face<'_>, text: &str, font_size: f32) -> PreparedText {
    let glyph_runs = shape_simple(face, text);
    let mut glyph_map = HashMap::<u16, usize>::new();
    let mut slug_glyphs = Vec::new();

    for placement in &glyph_runs {
        let key = placement.glyph_id.0;
        if glyph_map.contains_key(&key) {
            continue;
        }
        if let Some((curves, bounds)) = extract_curves(face, placement.glyph_id) {
            let bands = build_bands(&curves, &bounds, DEFAULT_BAND_COUNT);
            let slug_glyph = SlugGlyph {
                glyph_id: key,
                curves,
                bands,
                bounds,
            };
            glyph_map.insert(key, slug_glyphs.len());
            slug_glyphs.push(slug_glyph);
        }
    }

    let packed = pack_glyph_data(&slug_glyphs);
    let units_per_em = face.units_per_em() as f32;
    let scale = if units_per_em > 0.0 {
        font_size / units_per_em
    } else {
        1.0
    };

    let mut verts = Vec::<f32>::new();
    let mut indices = Vec::<u32>::new();
    let mut cursor_x = 0.0f32;
    let mut quad_idx = 0u32;

    for placement in glyph_runs {
        let glyph_index = match glyph_map.get(&placement.glyph_id.0) {
            Some(idx) => *idx,
            None => {
                cursor_x += placement.x_advance;
                continue;
            }
        };
        let glyph = &slug_glyphs[glyph_index];
        let band_info = packed
            .glyph_band_info
            .get(glyph_index)
            .copied()
            .unwrap_or(GlyphBandInfo {
                glyph_loc_x: 0,
                glyph_loc_y: 0,
            });

        let bounds = glyph.bounds;
        let w = bounds.x_max - bounds.x_min;
        let h = bounds.y_max - bounds.y_min;

        let ox = (cursor_x + placement.x_offset) * scale;
        let oy = placement.y_offset * scale;
        let x0 = ox + bounds.x_min * scale;
        let y0 = oy + bounds.y_min * scale;
        let x1 = ox + bounds.x_max * scale;
        let y1 = oy + bounds.y_max * scale;

        let band_scale_x = if w.abs() > f32::EPSILON {
            glyph.bands.v_band_count as f32 / w
        } else {
            0.0
        };
        let band_scale_y = if h.abs() > f32::EPSILON {
            glyph.bands.h_band_count as f32 / h
        } else {
            0.0
        };
        let band_offset_x = -bounds.x_min * band_scale_x;
        let band_offset_y = -bounds.y_min * band_scale_y;

        let glyph_loc =
            pack_u32_as_f32(((band_info.glyph_loc_y as u32) << 16) | band_info.glyph_loc_x as u32);
        let band_max_x = glyph.bands.v_band_count.saturating_sub(1) as u32;
        let band_max_y = glyph.bands.h_band_count.saturating_sub(1) as u32;
        let band_max = pack_u32_as_f32((band_max_y << 16) | band_max_x);
        let inv_scale = if scale.abs() > f32::EPSILON {
            1.0 / scale
        } else {
            1.0
        };

        let corners = [
            (x0, y0, -1.0f32, -1.0f32, bounds.x_min, bounds.y_min),
            (x1, y0, 1.0f32, -1.0f32, bounds.x_max, bounds.y_min),
            (x1, y1, 1.0f32, 1.0f32, bounds.x_max, bounds.y_max),
            (x0, y1, -1.0f32, 1.0f32, bounds.x_min, bounds.y_max),
        ];

        for (px, py, nx, ny, ex, ey) in corners {
            verts.extend_from_slice(&[
                // pos
                px,
                py,
                nx,
                ny,
                // tex
                ex,
                ey,
                glyph_loc,
                band_max,
                // jdc
                inv_scale,
                0.0,
                0.0,
                inv_scale,
                // bnd
                band_scale_x,
                band_scale_y,
                band_offset_x,
                band_offset_y,
            ]);
        }

        let base = quad_idx * 4;
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        quad_idx += 1;
        cursor_x += placement.x_advance;
    }

    PreparedText {
        slug_glyphs,
        vertices: verts,
        indices,
        curve_tex_data: packed.curve_tex_data,
        band_tex_data: packed.band_tex_data,
        curve_tex_height: packed.curve_tex_height,
        band_tex_height: packed.band_tex_height,
        total_advance: cursor_x,
    }
}
