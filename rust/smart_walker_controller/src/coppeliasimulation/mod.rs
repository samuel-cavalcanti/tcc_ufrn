mod simple_scene_with_easy_obstacles;

mod coppeliasim_sensors;
mod target;

pub use simple_scene_with_easy_obstacles::SimpleSceneWithEasyObstacles;

pub struct Image {
    pub data: Vec<u8>,
    pub resolution: Vec<i64>,
}
