use zmq_remote_api::sim::Sim;
use zmq_remote_api::{serde_json, RemoteAPIError};

pub struct Gyroscope {
    pub signal: String,
}

impl Gyroscope {
    pub fn read<S: Sim>(&self, coppeliasim: &S) -> Result<Vec<f32>, RemoteAPIError> {
        let data_string = coppeliasim.sim_get_string_signal(self.signal.clone())?;

        let json: serde_json::Value = serde_json::from_str(data_string.as_str()).unwrap();

        Ok(serde_json::from_value(json).unwrap())
    }
}
