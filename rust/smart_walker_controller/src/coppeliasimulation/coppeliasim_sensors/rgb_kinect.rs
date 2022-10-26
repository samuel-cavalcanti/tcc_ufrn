use crate::coppeliasimulation::Image;
use zmq_remote_api::sim::Sim;
use zmq_remote_api::{serde_json, sim, RemoteAPIError};

pub struct RgbKinect {
    pub handle: i64,
}

impl RgbKinect {
    pub fn read(&self, coppeliasim: &Sim) -> Result<Image, RemoteAPIError> {
        let (image, resolution) =
            coppeliasim.get_vision_sensor_img(self.handle, None, None, None, None)?;

        Ok(Image {
            data: image,
            resolution,
        })
    }
}
