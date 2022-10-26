use super::deep_learning_model::Model;

pub struct NNController {
    model: Model,
}

impl NNController {
    pub fn step(&mut self, target_pos: Vec<f64>, _current_theta: f64) -> Vec<f64> {
        let distance = (0..2).map(|i| target_pos[i].powi(2)).sum::<f64>().sqrt();

        let target_theta= f64::atan2(target_pos[1], target_pos[0]);

        let desired_vel = Self::get_desire_linear_vel(distance);

        let desired_theta = Self::get_desire_angular_vel(target_theta);

        let vel = self.model.predict(&[desired_vel, desired_theta]);

        vel
    }

    fn get_desire_linear_vel(distance: f64) -> f64 {
        if distance < 0.3 {
            return 0.1;
        }

        if distance > 1.0 {
            return 0.5;
        }

        5.0 / 7.0 * distance - 3.0 / 14.0
    }

    fn get_desire_angular_vel(theta: f64) -> f64 {
       
        
        let three_degree_in_rads =  0.05235987755982989;
        let thirty_degree_in_rads = 0.5235987755982988;
        let one_degree_in_rads = 0.017453292519943295;
        let five_degree_in_rads = 0.08726646259971647;

        if theta < three_degree_in_rads {
            return one_degree_in_rads;
        }

        if theta > thirty_degree_in_rads {
            return five_degree_in_rads;
        }

         // 5/27*x - 5/9 = y in degrees
        0.0032320912073969064 * theta - 0.00969627362219072
    }
}
