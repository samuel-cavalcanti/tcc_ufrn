use crate::coppeliasimulation::Image;
use zmq_remote_api::sim::Sim;
use zmq_remote_api::RemoteAPIError;

pub struct DepthKinect {
    pub handle: i64,
}

impl DepthKinect {
    pub fn read<S: Sim>(&self, coppeliasim: &S) -> Result<Image, RemoteAPIError> {
        let (image, resolution) =
            coppeliasim.sim_get_vision_sensor_depth(self.handle, None, None, None)?;

        Ok(Image {
            data: image,
            resolution,
        })
    }
}
