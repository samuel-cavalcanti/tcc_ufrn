use std::collections::VecDeque;
use tch::{
    nn::{self, Module, OptimizerConfig},
    Reduction,
};

pub struct Model {
    model: tch::nn::Sequential,
    var_store: tch::nn::VarStore,
    memory: VecDeque<(tch::Tensor, tch::Tensor)>,
    memory_capacity: usize,
}

impl Model {
    pub fn new(input_dim: i64, output_dim: i64) -> Model {
        let var = tch::nn::VarStore::new(tch::Device::cuda_if_available());
        let model = build_model(&var.root(), input_dim, output_dim);
        let memory_capacity = 1_000;

        Model {
            model,
            memory_capacity,
            var_store: var,
            memory: VecDeque::with_capacity(memory_capacity),
        }
    }

    pub fn predict(&self, input: &[f64]) -> Vec<f64> {
        let tensor_x = tch::Tensor::of_slice(input);
        self.model.forward(&tensor_x).into()
    }

    pub fn save_model(&self, path: String) {
        self.var_store.save(path).unwrap();
    }

    pub fn load_model(&mut self, path: String, input_dim: i64, output_dim: i64) {
        self.var_store.load(path).unwrap();
        self.model = build_model(&self.var_store.root(), input_dim, output_dim);
    }

    pub fn train(&mut self, x_true: &[f64], y_true: &[f64]) {
        let tensor_x = tch::Tensor::of_slice(x_true);
        let tensor_y = tch::Tensor::of_slice(y_true);

        if self.memory.len() == self.memory_capacity {
            self.memory.pop_back();
        }
        self.memory.push_front((tensor_x, tensor_y));

        self._train();
    }

    fn _train(&mut self) {
        if self.memory.len() < 20 {
            return;
        }

        let mut optimize = nn::RmsProp::default().build(&self.var_store, 1e-3).unwrap();

        for epoch in 0..5 {
            let mut last_loss = -1.0;
            for (tensor_x, tensor_y) in self.memory.iter() {
                let predict = self.model.forward(&tensor_x);
                let loss = predict.mse_loss(&tensor_y, Reduction::Sum);

                optimize.backward_step(&loss);
                last_loss = f64::from(loss);
            }
            println!("Epoch: {} loss: {:.3} ", epoch + 1, last_loss);
        }
    }
}

fn build_model(vs_path: &tch::nn::Path, input_dim: i64, output_dim: i64) -> tch::nn::Sequential {
    tch::nn::seq()
        .add(tch::nn::linear(
            vs_path / "layer_1",
            input_dim,
            32,
            Default::default(),
        ))
        .add_fn(|t| t.relu())
        .add(tch::nn::linear(
            vs_path / "layer_2",
            32,
            16,
            Default::default(),
        ))
        .add_fn(|t| t.relu())
        .add(tch::nn::linear(
            vs_path / "layer_3",
            16,
            output_dim,
            Default::default(),
        ))
}
