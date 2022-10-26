use crate::coppeliasimulation::Image;
use zmq_remote_api::sim::Sim;
use zmq_remote_api::{serde_json, sim, RemoteAPIError};

pub struct DepthKinect {
    pub handle: i64,
}

impl DepthKinect {
    pub fn read(&self, coppeliasim: &Sim) -> Result<Image, RemoteAPIError> {
        let (image, resolution) =
            coppeliasim.get_vision_sensor_depth(self.handle, None, None, None)?;

        Ok(Image {
            data: image,
            resolution,
        })
    }
}
