use crate::{controller, coppeliasimulation::Coppeliasim};
use tch::{nn::ModuleT, Tensor};
pub fn run() {
    let mut sim = Coppeliasim::new();

    let kinematic = load_kinematic_model();
    let mut controller = controller::FredericoController::new();

    sim.enable_step();
    sim.start_simulation();

    let mut robot_pos = vec![0.0, 0.0, 0.0];
    let mut target_pos = vec![0.0, 0.0];
    let mut kinematic_input = vec![0.0, 0.0, 0.0];

    let limit = vec![0.16133267, 0.04307239, 12.0_f32.to_radians()];

    let mut target_getted = 0;

    while target_getted < 5 {
        let theta = sim.get_orientation_z() as f32;
        let robot_pos_sim = sim.get_robot_pos();
        let target_pos_sim = sim.get_target_pos();

        robot_pos[0] = robot_pos_sim[0] as f32;
        robot_pos[1] = robot_pos_sim[1] as f32;
        robot_pos[2] = theta;

        target_pos[0] = target_pos_sim[0] as f32;
        target_pos[1] = target_pos_sim[1] as f32;

        let vel_robot = controller.step(&robot_pos, &target_pos);

        let cos_theta = theta.cos();
        let sin_theta = theta.sin();

        /*
             q_r = R(theta)q

            |vx_r|  =   | cos(theta) sen(theta)| |vx|
            |vy_r|  =   |-sin(theta) cos(theta)| |vy|

            mudando a referência da velocidade do referencial global (q) para o referencial do robô (q_r)

        */
        kinematic_input[0] = vel_robot[0] * cos_theta + vel_robot[1] * sin_theta;
        kinematic_input[1] = -vel_robot[0] * sin_theta + vel_robot[1] * cos_theta;
        kinematic_input[2] = vel_robot[2];

        for i in 0..3 {
            if kinematic_input[i].abs() > limit[i] {
                kinematic_input[i] = kinematic_input[i] / kinematic_input[i] * limit[i];
            }
        }

        let tensor = Tensor::of_slice(&kinematic_input).reshape(&[1, -1]);

        let vel_wheels = kinematic.forward_t(&tensor, false);
        let vel_wheels = Vec::<f64>::from(vel_wheels);

        let message = format!(
            "vel x:{:0.4} vel y:{:0.4} theta {:0.4}\nkinematic x:{:0.4} kinematic y:{:0.4} kinematic {:0.4}\n wheels: left {:0.4} right: {:0.4}",
            vel_robot[0],vel_robot[1],vel_robot[2].to_degrees(),
            kinematic_input[0],kinematic_input[1],kinematic_input[2].to_degrees(),
            vel_wheels[0], vel_wheels[1]
        );

        println!("{}", message);

        sim.set_velocity(vel_wheels[0], vel_wheels[1]);

        for _ in 0..5 {
            sim.step();
        }

        if is_goal(&target_pos, &robot_pos) {
            sim.update_target();
            target_getted += 1;
        }
    }

    sim.stop_simulation();
}

fn is_goal(target_pos: &Vec<f32>, robot_pos: &Vec<f32>) -> bool {
    let thirty_centimeters = 0.3;
    let distance = [0, 1]
        .iter()
        .map(|&i| (target_pos[i] - robot_pos[i]).powi(2))
        .sum::<f32>()
        .sqrt();

    distance < thirty_centimeters
}

fn load_kinematic_model() -> tch::CModule {
    tch::CModule::load("assets/kinematic.pt").unwrap()
}
