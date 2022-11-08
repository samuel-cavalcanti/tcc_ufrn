use crate::coppeliasimulation::coppeliasim_sensors::{
    Accelerometer, DepthKinect, Gyroscope, RgbKinect,
};
use crate::coppeliasimulation::target::Target;
use crate::coppeliasimulation::Image;
use std::rc::Rc;
use zmq_remote_api::sim::Sim;
use zmq_remote_api::{sim, RemoteAPIError, RemoteApiClient, RemoteApiClientParams};

pub struct Coppeliasim {
    client: Rc<RemoteApiClient>,
    sim: Sim,
    accelerometer: Accelerometer,
    gyroscope: Gyroscope,
    motors_handle: Motors,
    rgb_kinect: RgbKinect,
    depth_kinect: DepthKinect,
    target: Target,
    robot_handle: i64,
}

struct Motors {
    left_handle: i64,
    right_handle: i64,
}

struct Kinect {
    rgb_handle: i64,
    depth_handle: i64,
}

impl Coppeliasim {
    pub fn new() -> Coppeliasim {
        Coppeliasim::connect().unwrap()
    }

    fn connect() -> Result<Coppeliasim, RemoteAPIError> {
        let client: RemoteApiClient = zmq_remote_api::RemoteApiClient::new(RemoteApiClientParams {
            host: "localhost".to_string(),
            ..RemoteApiClientParams::default()
        })
        .expect("unable to connect to localhost");

        let client = Rc::new(client);

        let sim: Sim = Sim::new(client.clone());

        let mass_block = sim.get_object(String::from("./Accelerometer/mass"), None)?;
        let force_sensor = sim.get_object(String::from("./Accelerometer/forceSensor"), None)?;

        let right_motor = sim.get_object(String::from("./DynamicRightJoint"), None)?;
        let left_motor = sim.get_object(String::from("./DynamicLeftJoint"), None)?;

        let rgb_kinect_handle = sim.get_object(String::from("./kinect/rgb"), None)?;
        let kinect_depth = sim.get_object(String::from("./kinect/depth"), None)?;

        let target_handle = sim.get_object(String::from("./target"), None)?;

        let robot_handle = sim.get_object(String::from("./metade_lateral_r"), None)?;

        let mut target = Target::new(target_handle);
        target.move_target(&sim)?;

        let mass = sim.get_object_float_param(mass_block, sim::SHAPEFLOATPARAM_MASS)?;

        let acc = Accelerometer {
            mass,
            handle: force_sensor,
        };

        let motors = Motors {
            left_handle: left_motor,
            right_handle: right_motor,
        };

        let coppelia_sim: Coppeliasim = Coppeliasim {
            sim,
            client,
            target,
            accelerometer: acc,
            motors_handle: motors,
            robot_handle,
            rgb_kinect: RgbKinect {
                handle: rgb_kinect_handle,
            },
            depth_kinect: DepthKinect {
                handle: kinect_depth,
            },
            gyroscope: Gyroscope {
                signal: String::from("GyroSensor"),
            },
        };

        Ok(coppelia_sim)
    }

    pub fn enable_step(&self) {
        self.client.set_stepping(true).unwrap();
    }

    pub fn step(&self) {
        self.client.step(true).unwrap();
    }

    pub fn get_target_distance(&self) -> Vec<f64> {
        self.sim
            .get_object_position(self.target.handle, self.robot_handle)
            .unwrap()
    }

    pub fn get_target_pos(&self) -> Vec<f64> {
        self.sim
            .get_object_position(self.target.handle, sim::HANDLE_WORLD)
            .unwrap()
    }

    pub fn get_robot_pos(&self) -> Vec<f64> {
        self.sim
            .get_object_position(self.robot_handle, sim::HANDLE_WORLD)
            .unwrap()
    }

    pub fn update_target(&mut self) {
        self.target.move_target(&self.sim).unwrap();
    }

    pub fn get_acceleration(&self) -> Vec<f32> {
        self.accelerometer.read(&self.sim).unwrap()
    }

    pub fn get_gyroscope_data(&self) -> Vec<f32> {
        self.gyroscope.read(&self.sim).unwrap()
    }

    pub fn get_rgb_kinect(&self) -> Image {
        self.rgb_kinect.read(&self.sim).unwrap()
    }

    pub fn get_depth_kinect(&self) -> Image {
        self.depth_kinect.read(&self.sim).unwrap()
    }

    pub fn get_time(&self) -> f64 {
        self.sim.get_simulation_time().unwrap()
    }
    pub fn get_orientation_z(&self) -> f64 {
        let ori = self
            .sim
            .get_object_orientation(self.robot_handle, sim::HANDLE_WORLD)
            .unwrap();
        ori[2]
    }

    pub fn start_simulation(&self) {
        self.sim.start_simulation().unwrap();
    }

    pub fn stop_simulation(&self) {
        self.sim.stop_simulation().unwrap();
    }

    pub fn set_velocity(&self, left: f64, right: f64) {
        self.sim
            .set_joint_target_velocity(self.motors_handle.left_handle, left as f64, None, None)
            .unwrap();
        self.sim
            .set_joint_target_velocity(self.motors_handle.right_handle, right as f64, None, None)
            .unwrap();
    }
}
