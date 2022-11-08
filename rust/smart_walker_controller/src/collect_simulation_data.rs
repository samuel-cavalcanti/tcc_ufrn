use crate::{controller, coppeliasimulation::Coppeliasim};

pub fn collect_data() {
    let mut sim = Coppeliasim::new();

    sim.enable_step();

    sim.start_simulation();
    // Delay to read all sensors is
    // Max: 0.42951416969299316 Min: 0.2791721820831299 Mean: 0.3387012581030528

    let mut csv_writer = csv::Writer::from_path("data.csv").unwrap();

    csv_writer
        .write_record(&["target_x", "target_y", "theta", "motor_left", "motor_right"])
        .unwrap();

    loop {
        let target_pos = sim.get_target_distance();
        let ori = sim.get_orientation_z();

        let distance = target_pos[0..2]
            .iter()
            .map(|e| e.powi(2))
            .sum::<f64>()
            .sqrt();

        println!("distance: {}", distance);
        if distance < 0.3 {
            sim.update_target();
        }

        let velocity = controller::ManualController::get_velocity();

        match velocity {
            Some(vel) => {
                sim.set_velocity(vel[0], vel[1]);

                csv_writer
                    .write_record(&[
                        target_pos[0].to_string(),
                        target_pos[1].to_string(),
                        ori.to_string(),
                        vel[0].to_string(),
                        vel[1].to_string(),
                    ])
                    .unwrap();
            }
            None => break,
        }

        for _ in 0..5 {
            sim.step();
        }
    }
    sim.stop_simulation();

    csv_writer.flush().unwrap();
}
