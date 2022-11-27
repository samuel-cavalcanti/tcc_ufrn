from deep_learning_tests.torch_tests import test_torch_model_sim_data
from datasets import load_sim_data_rotate, SIM_DATA_ROTATE_PATH, load_sim_data_dataset
from deep_learning_tests.torch_tests import kinematic_model
import collect_data
import numpy as np
import torch
import timeit


def run_benchmark() -> None:
    model  = kinematic_model.load_kinematic().build_linear()
    random_size = 1_000_000
    random_samples = np.random.sample((random_size,3)).astype(np.float32)
    tensor_random = torch.from_numpy(random_samples)



    def benchmark():
        for sample in tensor_random:
            model.forward(sample)

    runs = 1
    print('start benchmark')
    duration = timeit.Timer(benchmark).timeit(runs)
    print('finished benchmark')
    print(f'duration: {duration:0.4f} seconds')
    avg = duration/random_size
    print(f'On average it took {avg} seconds')

def main() -> None:

    test_torch_model_sim_data.train_kinematic_model()

    model = kinematic_model.load_kinematic()
    
    for name, param in model.named_parameters():
        print(f'{name}:{param.data}')
    
    test_torch_model_sim_data.evaluate()
 
    # collect_data.main()

    pass


if __name__ == '__main__':
    main()
