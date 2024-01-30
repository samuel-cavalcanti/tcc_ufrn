use crate::coppeliasimulation::Image;
use zmq_remote_api::sim::Sim;
use zmq_remote_api::RemoteAPIError;

pub struct RgbKinect {
    pub handle: i64,
}

impl RgbKinect {
    pub fn read<S: Sim>(&self, coppeliasim: &S) -> Result<Image, RemoteAPIError> {
        let (image, resolution) =
            coppeliasim.sim_get_vision_sensor_img(self.handle, None, None, None, None)?;

        Ok(Image {
            data: image,
            resolution,
        })
    }
}
