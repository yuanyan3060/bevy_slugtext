use bevy::prelude::*;
use bevy_slugtext::prelude::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(SlugTextPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, update_text)
        .run();
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn(Camera3d::default()).insert(
        Transform::from_xyz(-0.5, -1.0, 5.0).looking_at(Vec3::new(-0.24, -0.5, -0.2), Vec3::Y),
    );

    commands.spawn((
        TextMesh {
            text: "hello world".to_string(),
            font: asset_server.load("fonts/Inter.ttf"),
            color: Color::Srgba(Srgba::WHITE),
            size: 1.0,
            ..Default::default()
        },
        Transform::from_xyz(-2.5, 0.0, 0.0),
    ));
}

fn update_text(
    mut value: Local<i32>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut query: Query<&mut TextMesh>,
) {
    if keyboard_input.just_pressed(KeyCode::KeyU) {
        *value += 1;
        for mut text in query.iter_mut() {
            text.text = value.to_string();
        }
    } else if keyboard_input.just_pressed(KeyCode::KeyD) {
        *value -= 1;
        for mut text in query.iter_mut() {
            text.text = value.to_string();
        }
    }
}

