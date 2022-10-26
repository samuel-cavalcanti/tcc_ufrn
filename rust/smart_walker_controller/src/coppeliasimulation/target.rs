use rand::Rng;
use zmq_remote_api::sim::Sim;
use zmq_remote_api::{sim, RemoteAPIError};

pub struct Target {
    pub handle: i64,
    positions: Vec<Vec<f64>>,
    current_index: usize,
}

impl Target {
    pub fn new(handle: i64) -> Target {
        let mut rng = rand::thread_rng();

        let positions = vec![
            vec![1.810547113418579, -1.9869998693466187, 0.2530002295970917],
            vec![-0.15445302426815033, -1.960999608039856, 0.2530002295970917],
            vec![-0.14445288479328156, -0.9789999127388, 0.2530002295970917],
            vec![-0.1354527771472931, 0.96200031042099, 0.2530002295970917],
            vec![-0.144452765583992, 1.948000192642212, 0.2530002295970917],
            vec![1.7815473079681396, 1.9449999332427979, 0.2530002295970917],
            vec![
                1.7695471048355103,
                -0.034000199288129807,
                0.2530002295970917,
            ],
            vec![
                0.8573786616325378,
                -0.014516029506921768,
                0.2530002295970917,
            ],
            vec![
                -0.15445293486118317,
                -0.00700022280216217,
                0.2530002295970917,
            ],
            vec![-2.0844528675079346, -0.9789997935295105, 0.2530002295970917],
            vec![-2.0794517993927, -1.9540002346038818, 0.2530002295970917],
            vec![-2.093451738357544, 0.006999874487519264, 0.2530002295970917],
            vec![-2.0964515209198, 1.9259999990463257, 0.2530002295970917],
            vec![-1.089451551437378, 1.9279992580413818, 0.2530002295970917],
            vec![-2.0504512786865234, 0.9569994807243347, 0.2530002295970917],
            vec![0.8035488128662109, 1.9239994287490845, 0.2530002295970917],
            vec![1.8005484342575073, 0.9539993405342102, 0.2530002295970917],
            vec![1.7905484437942505, -0.9860007762908936, 0.2530002295970917],
        ];

        let current = rng.gen_range(0..positions.len());

        Target {
            positions,
            handle,
            current_index: current,
        }
    }

    pub fn move_target(&mut self, sim: &Sim) -> Result<(), RemoteAPIError> {
        self.update_index();
        self.update_position(sim)?;
        Ok(())
    }

    fn update_index(&mut self) {
        let mut rng = rand::thread_rng();

        let mut new_index = rng.gen_range(0..self.positions.len());

        while new_index == self.current_index {
            new_index = rng.gen_range(0..self.positions.len());
        }

        self.current_index = new_index;
    }

    fn update_position(&self, sim: &Sim) -> Result<(), RemoteAPIError> {
        let position = self.positions[self.current_index].clone();
        sim.set_object_position(self.handle, sim::HANDLE_WORLD, position)?;

        Ok(())
    }
}
