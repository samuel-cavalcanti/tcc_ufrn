struct PID {
    k_p: f32,
    k_i: f32,
    k_d: f32,
    last_error: f32,
    integral_value: f32,
}

impl PID {
    pub fn new(k_p: f32, k_i: f32, k_d: f32) -> PID {
        PID {
            k_p,
            k_i,
            k_d,
            last_error: 0.0,
            integral_value: 0.0,
        }
    }

    pub fn step(&mut self, current_value: f32, desired_value: f32) -> f32 {
        let error = desired_value - current_value;
        self.integral_value += error;

        self.integral_value = if self.integral_value > 5.0 {
            5.0
        } else {
            self.integral_value
        };
        self.integral_value = if self.integral_value < -5.0 {
            -5.0
        } else {
            self.integral_value
        };

        let p = error * self.k_p;
        let i = self.integral_value * self.k_i;
        let d = (error - self.last_error) * self.k_d;

        self.last_error = error;

        p + i + d
    }
}

pub struct FredericoController {
    pid_pos: PID,
    pid_ori: PID,
}

impl FredericoController {
    pub fn new() -> FredericoController {
        let k_p_pos = 0.9145;
        let k_p_ori = 0.4;
        FredericoController {
            pid_pos: PID::new(k_p_pos, k_p_pos / 1.9079, k_p_pos / 0.085),
            pid_ori: PID::new(k_p_ori, 0.15, k_p_ori * 0.0474),
        }
    }

    pub fn step(&mut self, input: &Vec<f32>, target: &Vec<f32>) -> (f32, f32) {
        /*
           target[0] is the distance x between the robot and target
           target[1] is the distance y between the robot and target
           input[2] is the orientation of robot
           input[3] is the current simulation time
        */

        let mut delta: Vec<f32> = Vec::with_capacity(2); // (0..2).map(|i| input[i] - target[i]).collect();
        let mut l = 0.0;
        for i in 0..2 {
            let diff = target[i];
            delta.push(diff);
            l += diff.powi(2);
        }

        let phi = f32::atan2(delta[1], delta[0]);

        let current_theta = input[2];

        let delta_phi = phi - current_theta;

        let velocity_l = self.pid_pos.step(l * delta_phi.cos(), 0.0);
        let angular_velocity = self.pid_ori.step(delta_phi, 0.0);

        let velocity_linear_x = velocity_l * current_theta.cos();
        let velocity_linear_y = velocity_l * current_theta.sin();

        let left_wheel = velocity_linear_x + angular_velocity;
        let right_wheel = velocity_linear_y - angular_velocity;

        (left_wheel, right_wheel)
    }
}

