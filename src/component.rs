use bevy::prelude::*;
use bevy::render::render_resource::AsBindGroup;
use bevy::shader::ShaderRef;

use crate::slug;

#[derive(Component)]
#[require(Mesh3d)]
pub struct TextMesh {
    pub text: String,
    pub font: Handle<Font>,
    pub color: Color,
    pub bg_color: Color,
    pub size: f32,
}

impl Default for TextMesh {
    fn default() -> Self {
        Self {
            text: "".to_string(),
            font: Handle::default(),
            color: Color::BLACK,
            bg_color: Color::BLACK.with_alpha(0.0),
            size: 1.0,
        }
    }
}

#[derive(Component)]
pub struct TextMeshComputed;

#[derive(Asset, AsBindGroup, Reflect, Debug, Clone, Default)]
pub struct TextMaterial {
    #[texture(100, dimension = "2d", sample_type = "float")]
    pub curve_texture: Handle<Image>,

    #[texture(101, dimension = "2d", sample_type = "u_int")]
    pub band_texture: Handle<Image>,

    #[uniform(102)]
    pub color: LinearRgba,

    #[uniform(103)]
    pub bg_color: LinearRgba,
}

impl Material for TextMaterial {
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend
    }

    fn fragment_shader() -> ShaderRef {
        "embedded://bevy_slugtext/shaders/SlugPixel.wgsl".into()
    }

    fn vertex_shader() -> ShaderRef {
        "embedded://bevy_slugtext/shaders/SlugVertex.wgsl".into()
    }

    fn specialize(
        _pipeline: &bevy::pbr::MaterialPipeline,
        descriptor: &mut bevy::render::render_resource::RenderPipelineDescriptor,
        layout: &bevy::mesh::MeshVertexBufferLayoutRef,
        _key: bevy::pbr::MaterialPipelineKey<Self>,
    ) -> Result<(), bevy::render::render_resource::SpecializedMeshPipelineError> {
        let vertex_layout = layout.0.get_layout(&[
            Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
            Mesh::ATTRIBUTE_NORMAL.at_shader_location(1),
            slug::ATTRIBUTE_SLUG_POS.at_shader_location(8),
            slug::ATTRIBUTE_SLUG_TEX.at_shader_location(9),
            slug::ATTRIBUTE_SLUG_JAC.at_shader_location(10),
            slug::ATTRIBUTE_SLUG_BND.at_shader_location(11),
        ])?;

        descriptor.vertex.buffers = vec![vertex_layout];
        descriptor.primitive.cull_mode = None;

        Ok(())
    }
}
