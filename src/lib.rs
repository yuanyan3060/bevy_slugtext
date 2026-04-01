pub mod slug;
pub mod component;
pub mod systems;

use bevy::asset::embedded_asset;
use bevy::prelude::*;


use crate::component::TextMaterial;
use crate::systems::compute_mesh_and_material;

pub struct SlugTextPlugin;

impl Plugin for SlugTextPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shaders/SlugVertex.wgsl");
        embedded_asset!(app, "shaders/SlugPixel.wgsl");

        app.add_plugins(MaterialPlugin::<TextMaterial>::default())
            .add_systems(Update, compute_mesh_and_material);
    }
}

pub mod prelude {
    pub use super::SlugTextPlugin;
    pub use super::component::TextMaterial;
    pub use super::component::TextMesh;
}



