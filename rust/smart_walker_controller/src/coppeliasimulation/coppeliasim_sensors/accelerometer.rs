use zmq_remote_api::sim::Sim;
use zmq_remote_api::{sim, RemoteAPIError};

pub struct Accelerometer {
    pub mass: f64,
    pub handle: i64,
}

impl Accelerometer {
    pub fn read(&self, coppeliasim: &Sim) -> Result<Vec<f32>, RemoteAPIError> {
        let (result, forces, _torques) = coppeliasim.read_force_sensor(self.handle)?;

        if result == sim::RESULT_SUCCESS {
            let acceleration = forces
                .iter()
                .map(|force| (force / self.mass) as f32)
                .collect();
            Ok(acceleration)
        } else {
            Ok(vec![0.0, 0.0, 0.0])
        }
    }
}
