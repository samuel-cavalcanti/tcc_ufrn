use super::pid::PID;

pub struct FredericoController {
    pid_pos: PID,
    pid_ori: PID,
}

impl FredericoController {
    pub fn new() -> FredericoController {
        let k_p_pos = 0.05145;
        let k_p_ori = 0.2;
        FredericoController {
            pid_pos: PID::new(k_p_pos, k_p_pos / 2.9079, k_p_pos / 0.085),
            pid_ori: PID::new(k_p_ori, 0.015, 0.0),
        }
    }

    pub fn step(&mut self, input: &Vec<f32>, target: &Vec<f32>) -> Vec<f32> {
        /*
           target[0] pos x target
           target[1] pos y target

           input[0] robot pos x
           input[1] robot pos y
           input[2] is the orientation of robot

        */

        let mut robot_velocity = Vec::with_capacity(3);

        let delta_x = target[0] - input[0];
        let delta_y = target[1] - input[1];

        let delta_l = f32::sqrt(delta_x.powi(2) + delta_y.powi(2));
        let phi = f32::atan2(delta_y, delta_x);

        let theta = input[2];

        let delta_phi = phi - theta;

        let velocity_l = self.pid_pos.step(delta_l * delta_phi.cos(), 0.0);
        let angular_velocity = self.pid_ori.step(delta_phi, 0.0);

        let vel_x = velocity_l * theta.cos();
        let vel_y = velocity_l * theta.sin();

        robot_velocity.push(vel_x);
        robot_velocity.push(vel_y);
        robot_velocity.push(angular_velocity);

        robot_velocity
    }
}
