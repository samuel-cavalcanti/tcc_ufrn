mod coppeliasim;
mod coppeliasim_sensors;
mod target;

pub use coppeliasim::Coppeliasim;

pub struct Image {
    pub data: Vec<u8>,
    pub resolution: Vec<i64>,
}
