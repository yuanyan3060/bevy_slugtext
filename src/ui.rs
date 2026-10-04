//! Slug glyphs inside `bevy_ui`.
//!
//! A `Text` node marked [`SlugUiText`] keeps Bevy's layout (measuring,
//! wrapping, spans, fallback) and only swaps how its glyphs are drawn: the
//! glyph positions Bevy laid out are drawn as Slug quads in the UI pass, at
//! the node's stack index and inside its clip, and Bevy's atlas draw of the
//! same node is dropped.
//!
//! Bevy's layout says where a glyph sits and which atlas rect it was
//! rasterized into, not which glyph it is, so [`resolve_slug_glyphs`] reads
//! the glyph id and the font back out of the [`FontAtlasSet`]. A text with a
//! glyph that cannot be read back (a font that is not a [`Font`] asset, a
//! color glyph) keeps Bevy's atlas draw.
//!
//! Every glyph's curves live in one pair of textures ([`SlugGlyphStore`]),
//! in font units, so a glyph is stored once whatever its size and all UI
//! text draws from the same bind group.

use core::ops::Range;
use std::cmp::Ordering;

use bevy::asset::{embedded_asset, load_embedded_asset, RenderAssetUsages};
use bevy::ecs::system::{
    lifetimeless::{Read, SRes},
    SystemParamItem,
};
use bevy::math::{Affine2, FloatOrd};
use bevy::mesh::VertexBufferLayout;
use bevy::platform::collections::{HashMap, HashSet};
use bevy::prelude::*;
use bevy::render::{
    render_asset::RenderAssets,
    render_phase::*,
    render_resource::{
        binding_types::{texture_2d, uniform_buffer},
        *,
    },
    renderer::{RenderDevice, RenderQueue},
    sync_world::{MainEntity, TemporaryRenderEntity},
    texture::GpuImage,
    view::{ExtractedView, ViewUniform, ViewUniformOffset, ViewUniforms},
    Extract, ExtractSchedule, GpuResourceAppExt, Render, RenderApp, RenderStartup, RenderSystems,
};
use bevy::shader::Shader;
use bevy::text::{ComputedTextBlock, Font, FontAtlasKey, FontAtlasSet, TextLayoutInfo};
use bevy::ui::{
    CalculatedClip, ComputedNode, ComputedStackIndex, ComputedUiTargetCamera, UiGlobalTransform,
    UiSystems,
};
use bevy::ui_render::{
    stack_z_offsets, ExtractedUiItem, ExtractedUiNodes, RenderUiSystems, TransparentUi,
    UiCameraMap, UiCameraView,
};
use bytemuck::{Pod, Zeroable};

use crate::slug::{self, Bounds, QuadCurve, DEFAULT_BAND_COUNT, TEX_WIDTH};

/// How far a glyph's quad reaches past its outline, in physical pixels, so
/// the antialiased edge is not cut by the quad.
const DILATE_PX: f32 = 1.0;

pub struct SlugUiPlugin;

impl Plugin for SlugUiPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shaders/SlugUi.wgsl");

        app.init_resource::<SlugGlyphStore>()
            .init_resource::<SlugUiSettings>()
            .add_systems(PostUpdate, resolve_slug_glyphs.after(UiSystems::PostLayout));

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .add_render_command::<TransparentUi, DrawSlugUi>()
                .init_resource::<ExtractedSlugTexts>()
                .init_gpu_resource::<SlugUiMeta>()
                .init_gpu_resource::<SpecializedRenderPipelines<SlugUiPipeline>>()
                .add_systems(RenderStartup, init_slug_ui_pipeline)
                .add_systems(
                    ExtractSchedule,
                    extract_slug_texts.after(RenderUiSystems::ExtractText),
                )
                .add_systems(
                    Render,
                    (
                        queue_slug_texts.in_set(RenderSystems::Queue),
                        prepare_slug_texts.in_set(RenderSystems::PrepareBindGroups),
                    ),
                );
        }
    }
}

/// On a `Text` node: draw its glyphs with Slug instead of the font atlas.
#[derive(Component, Clone, Copy, Default)]
pub struct SlugUiText;

/// Turns the whole pass off (every text back on Bevy's atlas) without
/// touching the markers, for comparing the two.
#[derive(Resource, Clone, Copy)]
pub struct SlugUiSettings {
    pub enabled: bool,
}

impl Default for SlugUiSettings {
    fn default() -> Self {
        Self { enabled: true }
    }
}

// ---- The glyph store ----

/// Where a glyph's curves and bands sit in the store's textures.
#[derive(Clone, Copy, Debug)]
struct StoredGlyph {
    bounds: Bounds,
    /// The glyph's first band texel.
    loc: [u32; 2],
    band_max: [u32; 2],
    units_per_em: f32,
}

/// Every glyph any UI text has drawn, by font and glyph id, in two textures
/// that only grow. Curves are in font units, so size is not part of the key.
#[derive(Resource)]
pub struct SlugGlyphStore {
    /// `None` for a glyph with no outline (a space).
    glyphs: HashMap<(u32, u32, u16), Option<StoredGlyph>>,
    curve_data: Vec<f32>,
    curve_texels: usize,
    band_data: Vec<u32>,
    band_texels: usize,
    curve_image: Handle<Image>,
    band_image: Handle<Image>,
    dirty: bool,
}

impl FromWorld for SlugGlyphStore {
    fn from_world(world: &mut World) -> Self {
        let mut images = world.resource_mut::<Assets<Image>>();
        let curve_data = vec![0.0; TEX_WIDTH * 4];
        let band_data = vec![0; TEX_WIDTH * 4];
        Self {
            glyphs: HashMap::default(),
            curve_image: images.add(store_image(&curve_data, TextureFormat::Rgba32Float)),
            band_image: images.add(store_image(&band_data, TextureFormat::Rgba32Uint)),
            curve_data,
            curve_texels: 0,
            band_data,
            band_texels: 0,
            dirty: false,
        }
    }
}

fn store_image<T: Pod>(data: &[T], format: TextureFormat) -> Image {
    let mut image = Image::new(
        Extent3d {
            width: TEX_WIDTH as u32,
            height: (data.len() / (TEX_WIDTH * 4)) as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        bytemuck::cast_slice(data).to_vec(),
        format,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
    image
}

/// Whole rows of the store's texture that hold `texels`.
fn rows_len(texels: usize) -> usize {
    texels.div_ceil(TEX_WIDTH).max(1) * TEX_WIDTH * 4
}

impl SlugGlyphStore {
    /// Appends a glyph's curves and bands, in the layout `SlugRender` reads:
    /// two texels a curve, then per glyph a header texel per band (count,
    /// offset to its list) followed by the bands' curve lists.
    fn push(&mut self, curves: &[QuadCurve], bounds: &Bounds) -> [u32; 2] {
        // A curve's second texel is read at `x + 1` with no wrap. The width
        // is even and every curve takes two texels, so none straddles a row.
        let curve_start = self.curve_texels;
        self.curve_texels += curves.len() * 2;
        self.curve_data.resize(rows_len(self.curve_texels), 0.0);
        for (i, c) in curves.iter().enumerate() {
            let at = (curve_start + i * 2) * 4;
            self.curve_data[at..at + 6].copy_from_slice(&[c.p0x, c.p0y, c.p1x, c.p1y, c.p2x, c.p2y]);
        }

        let bands = slug::build_bands(curves, bounds, DEFAULT_BAND_COUNT);
        // Each band's curves from the far edge in, so the shader can stop at
        // the first one wholly behind the pixel.
        let sorted = |band: &slug::BandEntry, far: fn(&QuadCurve) -> f32| {
            let mut indices = band.curve_indices.clone();
            indices.sort_by(|&a, &b| {
                far(&curves[b]).partial_cmp(&far(&curves[a])).unwrap_or(Ordering::Equal)
            });
            indices
        };
        let lists: Vec<Vec<usize>> = bands
            .h_bands
            .iter()
            .map(|b| sorted(b, |c| c.p0x.max(c.p1x).max(c.p2x)))
            .chain(bands.v_bands.iter().map(|b| sorted(b, |c| c.p0y.max(c.p1y).max(c.p2y))))
            .collect();

        // The headers are read at `x + band` with no wrap: keep them on a row.
        let header = lists.len();
        let x = self.band_texels % TEX_WIDTH;
        if x + header > TEX_WIDTH {
            self.band_texels += TEX_WIDTH - x;
        }
        let start = self.band_texels;
        let total = header + lists.iter().map(Vec::len).sum::<usize>();
        self.band_texels = start + total;
        self.band_data.resize(rows_len(self.band_texels), 0);

        let mut offset = header;
        for (i, list) in lists.iter().enumerate() {
            let at = (start + i) * 4;
            self.band_data[at] = list.len() as u32;
            self.band_data[at + 1] = offset as u32;
            for (k, curve) in list.iter().enumerate() {
                let texel = curve_start + curve * 2;
                let at = (start + offset + k) * 4;
                self.band_data[at] = (texel % TEX_WIDTH) as u32;
                self.band_data[at + 1] = (texel / TEX_WIDTH) as u32;
            }
            offset += list.len();
        }

        self.dirty = true;
        [(start % TEX_WIDTH) as u32, (start / TEX_WIDTH) as u32]
    }

    /// The stored glyph, read out of `font` the first time it is asked for.
    /// The outer `None` is a font that will not parse.
    fn glyph(&mut self, key: (u32, u32, u16), font: &Font) -> Option<Option<StoredGlyph>> {
        if let Some(stored) = self.glyphs.get(&key) {
            return Some(*stored);
        }
        let face = ttf_parser::Face::parse(font.data.as_ref(), key.1).ok()?;
        let stored = slug::extract_curves(&face, ttf_parser::GlyphId(key.2))
            .filter(|(curves, _)| !curves.is_empty())
            .map(|(curves, bounds)| StoredGlyph {
                loc: self.push(&curves, &bounds),
                band_max: [DEFAULT_BAND_COUNT as u32 - 1; 2],
                bounds,
                units_per_em: face.units_per_em() as f32,
            });
        self.glyphs.insert(key, stored);
        Some(stored)
    }
}

// ---- Reading Bevy's layout back into glyphs ----

/// One laid-out glyph, in the text's own physical pixels.
#[derive(Clone, Copy)]
struct UiGlyph {
    /// The glyph's origin on the baseline.
    pen: Vec2,
    /// Pixels per font unit.
    scale: f32,
    section: usize,
    stored: StoredGlyph,
}

/// A [`SlugUiText`]'s glyphs as Slug draws them. Present only while every
/// glyph of the text could be read back; without it Bevy's atlas draws.
#[derive(Component, Default)]
pub struct SlugUiGlyphs(Vec<UiGlyph>);

/// Which glyph an atlas rect holds, read back out of the [`FontAtlasSet`].
#[derive(Default)]
struct AtlasIndex {
    by_rect: HashMap<(AssetId<Image>, [u32; 2]), (FontAtlasKey, u16)>,
    fonts: HashMap<u32, AssetId<Font>>,
    stale: bool,
}

#[allow(clippy::type_complexity)]
fn resolve_slug_glyphs(
    mut commands: Commands,
    atlases: Res<FontAtlasSet>,
    fonts: Res<Assets<Font>>,
    mut store: ResMut<SlugGlyphStore>,
    mut images: ResMut<Assets<Image>>,
    mut index: Local<AtlasIndex>,
    texts: Query<
        (Entity, &TextLayoutInfo),
        (With<SlugUiText>, Or<(Changed<TextLayoutInfo>, Added<SlugUiText>)>),
    >,
) {
    if atlases.is_changed() {
        index.stale = true;
    }
    if texts.is_empty() {
        return;
    }
    if index.stale {
        index.stale = false;
        index.by_rect.clear();
        for (key, font_atlases) in atlases.iter() {
            for atlas in font_atlases {
                for (glyph, location) in &atlas.glyph_to_atlas_index {
                    let Some(rect) = atlas.texture_atlas.textures.get(location.glyph_index) else {
                        continue;
                    };
                    index
                        .by_rect
                        .insert((atlas.texture.id(), rect.min.to_array()), (*key, glyph.glyph_id));
                }
            }
        }
    }

    let store = store.bypass_change_detection();
    for (entity, layout) in &texts {
        let mut glyphs = Vec::with_capacity(layout.glyphs.len());
        let mut whole = true;
        for g in &layout.glyphs {
            let info = &g.atlas_info;
            let rect_min = info.rect.min.as_uvec2().to_array();
            let Some(&(key, glyph_id)) = index.by_rect.get(&(info.texture, rect_min)) else {
                whole = false;
                break;
            };
            if !info.is_alpha_mask {
                whole = false;
                break;
            }
            if !index.fonts.contains_key(&key.id) {
                let found = fonts.iter().find(|(_, font)| font.data.id() as u32 == key.id);
                let Some((id, _)) = found else {
                    whole = false;
                    break;
                };
                index.fonts.insert(key.id, id);
            }
            let Some(stored) = fonts
                .get(index.fonts[&key.id])
                .and_then(|font| store.glyph((key.id, key.index, glyph_id), font))
            else {
                whole = false;
                break;
            };
            let Some(stored) = stored else {
                continue;
            };
            glyphs.push(UiGlyph {
                // Bevy places the bitmap's center: back to the origin.
                pen: g.position - info.rect.size() / 2.0 - info.offset,
                scale: f32::from_bits(key.font_size_bits) / stored.units_per_em,
                section: g.section_index,
                stored,
            });
        }
        if whole {
            commands.entity(entity).try_insert(SlugUiGlyphs(glyphs));
        } else {
            commands.entity(entity).try_remove::<SlugUiGlyphs>();
        }
    }

    if store.dirty {
        store.dirty = false;
        let curve = store_image(&store.curve_data, TextureFormat::Rgba32Float);
        let band = store_image(&store.band_data, TextureFormat::Rgba32Uint);
        let _ = images.insert(store.curve_image.id(), curve);
        let _ = images.insert(store.band_image.id(), band);
    }
}

// ---- Extraction ----

struct ExtractedSlugText {
    stack_index: u32,
    transform: Affine2,
    clip: Option<Rect>,
    extracted_camera_entity: Entity,
    main_entity: MainEntity,
    render_entity: Entity,
    glyphs: Range<usize>,
}

struct ExtractedSlugGlyph {
    glyph: UiGlyph,
    color: LinearRgba,
}

#[derive(Resource, Default)]
struct ExtractedSlugTexts {
    texts: Vec<ExtractedSlugText>,
    glyphs: Vec<ExtractedSlugGlyph>,
    images: Option<(AssetId<Image>, AssetId<Image>)>,
}

#[allow(clippy::type_complexity)]
fn extract_slug_texts(
    mut commands: Commands,
    mut extracted: ResMut<ExtractedSlugTexts>,
    mut extracted_uinodes: ResMut<ExtractedUiNodes>,
    settings: Extract<Res<SlugUiSettings>>,
    store: Extract<Res<SlugGlyphStore>>,
    texts: Extract<
        Query<(
            Entity,
            &SlugUiGlyphs,
            &ComputedNode,
            &ComputedStackIndex,
            &UiGlobalTransform,
            &InheritedVisibility,
            Option<&CalculatedClip>,
            &ComputedUiTargetCamera,
            &ComputedTextBlock,
            &TextColor,
        ), With<SlugUiText>>,
    >,
    colors: Extract<Query<&TextColor>>,
    camera_map: Extract<UiCameraMap>,
    mut taken: Local<HashSet<Entity>>,
) {
    extracted.texts.clear();
    extracted.glyphs.clear();
    extracted.images = Some((store.curve_image.id(), store.band_image.id()));
    if !settings.enabled {
        return;
    }
    taken.clear();

    let mut camera_mapper = camera_map.get_mapper();
    for (entity, glyphs, uinode, stack_index, transform, visibility, clip, camera, block, color) in
        &texts
    {
        // Taken even when hidden: Bevy skips the same texts itself.
        taken.insert(entity);
        if !visibility.get() || uinode.is_empty() {
            continue;
        }
        let Some(extracted_camera_entity) = camera_mapper.map(camera) else {
            continue;
        };

        let start = extracted.glyphs.len();
        let mut section = 0;
        let mut section_color = color.0.to_linear();
        for glyph in &glyphs.0 {
            if glyph.section != section {
                section = glyph.section;
                section_color = block
                    .entities()
                    .get(section)
                    .and_then(|t| colors.get(t.entity).ok())
                    .map(|c| c.0.to_linear())
                    .unwrap_or_default();
            }
            extracted.glyphs.push(ExtractedSlugGlyph { glyph: *glyph, color: section_color });
        }
        let end = extracted.glyphs.len();
        if start == end {
            continue;
        }

        extracted.texts.push(ExtractedSlugText {
            stack_index: stack_index.0,
            transform: Affine2::from(*transform)
                * Affine2::from_translation(uinode.content_box().min),
            clip: clip.map(|clip| clip.clip),
            extracted_camera_entity,
            main_entity: entity.into(),
            render_entity: commands.spawn(TemporaryRenderEntity).id(),
            glyphs: start..end,
        });
    }

    // Bevy has already extracted the same texts from its atlas: drop those.
    if !taken.is_empty() {
        extracted_uinodes.uinodes.retain(|node| {
            !(matches!(node.item, ExtractedUiItem::Glyphs { .. })
                && taken.contains(&node.main_entity.id()))
        });
    }
}

// ---- The pipeline ----

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SlugUiVertex {
    position: [f32; 2],
    /// The same point in the glyph's font units.
    em: [f32; 2],
    color: [f32; 4],
    /// Band scale (x, y) and offset (x, y), font units to band index.
    band: [f32; 4],
    /// The glyph's band texel (x, y) and its last band (x, y).
    glyph: [u32; 4],
}

#[derive(Component)]
struct SlugUiBatch {
    range: Range<u32>,
}

#[derive(Resource)]
struct SlugUiMeta {
    vertices: RawBufferVec<SlugUiVertex>,
    indices: RawBufferVec<u32>,
    view_bind_group: Option<BindGroup>,
    glyph_bind_group: Option<BindGroup>,
}

impl Default for SlugUiMeta {
    fn default() -> Self {
        Self {
            vertices: RawBufferVec::new(BufferUsages::VERTEX),
            indices: RawBufferVec::new(BufferUsages::INDEX),
            view_bind_group: None,
            glyph_bind_group: None,
        }
    }
}

#[derive(Resource)]
struct SlugUiPipeline {
    view_layout: BindGroupLayoutDescriptor,
    glyph_layout: BindGroupLayoutDescriptor,
    shader: Handle<Shader>,
}

fn init_slug_ui_pipeline(mut commands: Commands, asset_server: Res<AssetServer>) {
    let view_layout = BindGroupLayoutDescriptor::new(
        "slug_ui_view_layout",
        &BindGroupLayoutEntries::single(
            ShaderStages::VERTEX_FRAGMENT,
            uniform_buffer::<ViewUniform>(true),
        ),
    );
    let glyph_layout = BindGroupLayoutDescriptor::new(
        "slug_ui_glyph_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_2d(TextureSampleType::Uint),
            ),
        ),
    );
    commands.insert_resource(SlugUiPipeline {
        view_layout,
        glyph_layout,
        shader: load_embedded_asset!(asset_server.as_ref(), "shaders/SlugUi.wgsl"),
    });
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
struct SlugUiPipelineKey {
    target_format: TextureFormat,
}

impl SpecializedRenderPipeline for SlugUiPipeline {
    type Key = SlugUiPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let vertex_layout = VertexBufferLayout::from_vertex_formats(
            VertexStepMode::Vertex,
            vec![
                VertexFormat::Float32x2,
                VertexFormat::Float32x2,
                VertexFormat::Float32x4,
                VertexFormat::Float32x4,
                VertexFormat::Uint32x4,
            ],
        );
        RenderPipelineDescriptor {
            vertex: VertexState {
                shader: self.shader.clone(),
                buffers: vec![vertex_layout],
                ..default()
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                targets: vec![Some(ColorTargetState {
                    format: key.target_format,
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            layout: vec![self.view_layout.clone(), self.glyph_layout.clone()],
            label: Some("slug_ui_pipeline".into()),
            ..default()
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn queue_slug_texts(
    extracted: Res<ExtractedSlugTexts>,
    slug_pipeline: Res<SlugUiPipeline>,
    mut pipelines: ResMut<SpecializedRenderPipelines<SlugUiPipeline>>,
    mut transparent_render_phases: ResMut<ViewSortedRenderPhases<TransparentUi>>,
    mut render_views: Query<&UiCameraView, With<ExtractedView>>,
    camera_views: Query<&ExtractedView>,
    pipeline_cache: Res<PipelineCache>,
    draw_functions: Res<DrawFunctions<TransparentUi>>,
) {
    let draw_function = draw_functions.read().id::<DrawSlugUi>();
    for (index, text) in extracted.texts.iter().enumerate() {
        let Ok(default_camera_view) = render_views.get_mut(text.extracted_camera_entity) else {
            continue;
        };
        let Ok(view) = camera_views.get(default_camera_view.0) else {
            continue;
        };
        let Some(transparent_phase) = transparent_render_phases.get_mut(&view.retained_view_entity)
        else {
            continue;
        };
        let pipeline = pipelines.specialize(
            &pipeline_cache,
            &slug_pipeline,
            SlugUiPipelineKey { target_format: view.target_format },
        );
        transparent_phase.add_transient(TransparentUi {
            draw_function,
            pipeline,
            entity: (text.render_entity, text.main_entity),
            sort_key: FloatOrd(text.stack_index as f32 + stack_z_offsets::TEXT),
            batch_range: 0..0,
            extra_index: PhaseItemExtraIndex::None,
            index,
            indexed: true,
        });
    }
}

/// A glyph's quad: its corners clockwise from the top left, each with its
/// point in font units. `None` when the clip leaves nothing of it.
fn glyph_quad(
    glyph: &UiGlyph,
    transform: &Affine2,
    clip: Option<Rect>,
) -> Option<[([f32; 2], [f32; 2]); 4]> {
    let b = &glyph.stored.bounds;
    let s = glyph.scale;
    // Font units run up, pixels down.
    let min = glyph.pen + Vec2::new(b.x_min, -b.y_max) * s - DILATE_PX;
    let max = glyph.pen + Vec2::new(b.x_max, -b.y_min) * s + DILATE_PX;
    let em_min = Vec2::new(b.x_min - DILATE_PX / s, b.y_max + DILATE_PX / s);
    let em_max = Vec2::new(b.x_max + DILATE_PX / s, b.y_min - DILATE_PX / s);

    // Clipping an axis-aligned quad keeps it a quad; a turned one is drawn
    // whole, as Bevy's own text is.
    let upright = transform.matrix2.x_axis.y == 0.0 && transform.matrix2.y_axis.x == 0.0;
    let (min, max, em_min, em_max) = match clip {
        Some(clip) if upright => {
            let (a, b) = (transform.transform_point2(min), transform.transform_point2(max));
            let (lo, hi) = (a.min(b).max(clip.min), a.max(b).min(clip.max));
            if lo.cmpge(hi).any() {
                return None;
            }
            let span = b - a;
            let em_at = |p: Vec2| em_min + (p - a) / span * (em_max - em_min);
            let (ta, tb) = (
                Vec2::select(span.cmpge(Vec2::ZERO), lo, hi),
                Vec2::select(span.cmpge(Vec2::ZERO), hi, lo),
            );
            return Some([
                (ta.to_array(), em_at(ta).to_array()),
                ([tb.x, ta.y], em_at(Vec2::new(tb.x, ta.y)).to_array()),
                (tb.to_array(), em_at(tb).to_array()),
                ([ta.x, tb.y], em_at(Vec2::new(ta.x, tb.y)).to_array()),
            ]);
        }
        _ => (min, max, em_min, em_max),
    };
    let at = |p: Vec2| transform.transform_point2(p).to_array();
    Some([
        (at(min), em_min.to_array()),
        (at(Vec2::new(max.x, min.y)), [em_max.x, em_min.y]),
        (at(max), em_max.to_array()),
        (at(Vec2::new(min.x, max.y)), [em_min.x, em_max.y]),
    ])
}

#[allow(clippy::too_many_arguments)]
fn prepare_slug_texts(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    mut meta: ResMut<SlugUiMeta>,
    extracted: Res<ExtractedSlugTexts>,
    view_uniforms: Res<ViewUniforms>,
    slug_pipeline: Res<SlugUiPipeline>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    mut phases: ResMut<ViewSortedRenderPhases<TransparentUi>>,
) {
    let Some(view_binding) = view_uniforms.uniforms.binding() else {
        return;
    };
    meta.vertices.clear();
    meta.indices.clear();
    meta.view_bind_group = Some(render_device.create_bind_group(
        "slug_ui_view_bind_group",
        &pipeline_cache.get_bind_group_layout(&slug_pipeline.view_layout),
        &BindGroupEntries::single(view_binding),
    ));
    // The store's textures are replaced when a glyph is added, so the bind
    // group is made again each frame rather than tracked.
    meta.glyph_bind_group = extracted
        .images
        .and_then(|(curve, band)| Some((gpu_images.get(curve)?, gpu_images.get(band)?)))
        .map(|(curve, band)| {
            render_device.create_bind_group(
                "slug_ui_glyph_bind_group",
                &pipeline_cache.get_bind_group_layout(&slug_pipeline.glyph_layout),
                &BindGroupEntries::sequential((&curve.texture_view, &band.texture_view)),
            )
        });
    if meta.glyph_bind_group.is_none() {
        return;
    }

    let mut batches: Vec<(Entity, SlugUiBatch)> = Vec::new();
    let mut index_count = 0;
    let mut vertex_count = 0;

    for ui_phase in phases.values_mut() {
        // The item that opened the batch the next text joins. Texts that
        // follow each other in the phase draw as one.
        let mut open: Option<usize> = None;
        for item_index in 0..ui_phase.items.len() {
            let item = &ui_phase.items[item_index];
            let Some(text) =
                extracted.texts.get(item.index).filter(|t| item.entity() == t.render_entity)
            else {
                open = None;
                continue;
            };
            let opened = *open.get_or_insert_with(|| {
                batches.push((item.entity(), SlugUiBatch { range: index_count..index_count }));
                item_index
            });

            for extracted_glyph in &extracted.glyphs[text.glyphs.clone()] {
                let glyph = &extracted_glyph.glyph;
                let Some(quad) = glyph_quad(glyph, &text.transform, text.clip) else {
                    continue;
                };
                let b = &glyph.stored.bounds;
                let [max_x, max_y] = glyph.stored.band_max;
                let band_scale = Vec2::new(
                    (max_x + 1) as f32 / (b.x_max - b.x_min).max(f32::EPSILON),
                    (max_y + 1) as f32 / (b.y_max - b.y_min).max(f32::EPSILON),
                );
                let band = [
                    band_scale.x,
                    band_scale.y,
                    -b.x_min * band_scale.x,
                    -b.y_min * band_scale.y,
                ];
                let [loc_x, loc_y] = glyph.stored.loc;
                for (position, em) in quad {
                    meta.vertices.push(SlugUiVertex {
                        position,
                        em,
                        color: extracted_glyph.color.to_f32_array(),
                        band,
                        glyph: [loc_x, loc_y, max_x, max_y],
                    });
                }
                for i in [0, 1, 2, 0, 2, 3] {
                    meta.indices.push(vertex_count + i);
                }
                vertex_count += 4;
                index_count += 6;
            }

            batches.last_mut().unwrap().1.range.end = index_count;
            ui_phase.items[opened].batch_range_mut().end += 1;
        }
    }

    meta.vertices.write_buffer(&render_device, &render_queue);
    meta.indices.write_buffer(&render_device, &render_queue);
    commands.try_insert_batch(batches);
}

type DrawSlugUi = (SetItemPipeline, SetSlugUiViewBindGroup<0>, SetSlugUiGlyphBindGroup<1>, DrawSlugUiBatch);

struct SetSlugUiViewBindGroup<const I: usize>;
impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetSlugUiViewBindGroup<I> {
    type Param = SRes<SlugUiMeta>;
    type ViewQuery = Read<ViewUniformOffset>;
    type ItemQuery = ();

    fn render<'w>(
        _item: &P,
        view_uniform: &'w ViewUniformOffset,
        _entity: Option<()>,
        meta: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some(view_bind_group) = meta.into_inner().view_bind_group.as_ref() else {
            return RenderCommandResult::Failure("view_bind_group not available");
        };
        pass.set_bind_group(I, view_bind_group, &[view_uniform.offset]);
        RenderCommandResult::Success
    }
}

struct SetSlugUiGlyphBindGroup<const I: usize>;
impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetSlugUiGlyphBindGroup<I> {
    type Param = SRes<SlugUiMeta>;
    type ViewQuery = ();
    type ItemQuery = Read<SlugUiBatch>;

    fn render<'w>(
        _item: &P,
        _view: (),
        batch: Option<&'w SlugUiBatch>,
        meta: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        if batch.is_none() {
            return RenderCommandResult::Skip;
        }
        let Some(glyph_bind_group) = meta.into_inner().glyph_bind_group.as_ref() else {
            return RenderCommandResult::Skip;
        };
        pass.set_bind_group(I, glyph_bind_group, &[]);
        RenderCommandResult::Success
    }
}

struct DrawSlugUiBatch;
impl<P: PhaseItem> RenderCommand<P> for DrawSlugUiBatch {
    type Param = SRes<SlugUiMeta>;
    type ViewQuery = ();
    type ItemQuery = Read<SlugUiBatch>;

    fn render<'w>(
        _item: &P,
        _view: (),
        batch: Option<&'w SlugUiBatch>,
        meta: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some(batch) = batch else {
            return RenderCommandResult::Skip;
        };
        if batch.range.is_empty() {
            return RenderCommandResult::Skip;
        }
        let meta = meta.into_inner();
        let (Some(vertices), Some(indices)) = (meta.vertices.buffer(), meta.indices.buffer())
        else {
            return RenderCommandResult::Failure("missing slug ui buffers");
        };
        pass.set_vertex_buffer(0, vertices.slice(..));
        pass.set_index_buffer(indices.slice(..), IndexFormat::Uint32);
        pass.draw_indexed(batch.range.clone(), 0, 0..1);
        RenderCommandResult::Success
    }
}
