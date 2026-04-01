use bevy::prelude::*;

use crate::{component::{TextMaterial, TextMesh, TextMeshComputed}, slug};

pub fn compute_mesh_and_material(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut query: Query<
        (Entity, &TextMesh, &mut Mesh3d),
        Or<(Changed<TextMesh>, Without<TextMeshComputed>)>,
    >,
    mut text_materials: ResMut<Assets<TextMaterial>>,
    mut images: ResMut<Assets<Image>>,
    font_assets: Res<Assets<Font>>,
) {
    for (entity, text_mesh, mut mesh3d) in query.iter_mut() {
        let Some(font) = font_assets.get(&text_mesh.font) else {
            continue;
        };

        let Ok(face) = ttf_parser::Face::parse(&font.data, 0) else {
            continue;
        };

        let prepare_text = slug::prepare_text(&face, &text_mesh.text, text_mesh.size);
        mesh3d.0 = meshes.add(prepare_text.mesh());

        commands
            .entity(entity)
            .insert(TextMeshComputed)
            .try_remove::<MeshMaterial3d<TextMaterial>>()
            .insert(MeshMaterial3d(text_materials.add(TextMaterial {
                curve_texture: images.add(prepare_text.curve()),
                band_texture: images.add(prepare_text.band()),
                color: text_mesh.color.to_linear(),
                bg_color: text_mesh.bg_color.to_linear(),
            })));
    }
}