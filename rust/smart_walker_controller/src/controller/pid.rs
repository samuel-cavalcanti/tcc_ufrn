pub struct PID {
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

    fn windup_guard(&self, value: f32) -> f32 {
        if value > 5.0 {
            return 5.0;
        }

        if value < -5.0 {
            return -5.0;
        }

        value
    }

    pub fn step(&mut self, current_value: f32, desired_value: f32) -> f32 {
        let error = current_value - desired_value;

        self.integral_value = self.windup_guard(self.integral_value + error);
        let derivate_value = error - self.last_error;

        let p = error * self.k_p;
        let i = self.integral_value * self.k_i;
        let d = derivate_value * self.k_d;

        self.last_error = error;

        p + i + d
    }
}
