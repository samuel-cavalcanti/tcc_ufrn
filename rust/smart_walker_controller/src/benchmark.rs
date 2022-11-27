use crate::test_kinematic_model;
use tch::{self, nn::Module};
pub fn run() {
    let device = tch::Device::cuda_if_available();
    let model = test_kinematic_model::load_kinematic_model();

    let sample_size = 1_000_000;
    let random = tch::Tensor::rand(&[sample_size, 3], (tch::Kind::Float, device));

    let random: Vec<tch::Tensor> = (0..sample_size).map(|i| random.get(i)).collect();
   
    println!("Start benchmark");
    let instant = std::time::Instant::now();
   
    {
        for i in 0..(sample_size as usize) {
            let _ = model.forward(&random[i]);
        }
    }

    let duration = instant.elapsed();
    let duration_in_seconds = duration.as_secs_f64();
    let avg = duration_in_seconds  /(sample_size as f64);
    println!("duration: {:0.4?} seconds", duration_in_seconds);
    println!("On average it took {} seconds", avg);
}
