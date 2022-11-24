from deep_learning_tests.torch_tests import  test_torch_model_sim_data
from datasets import load_sim_data_rotate,SIM_DATA_ROTATE_PATH,load_sim_data_dataset
from deep_learning_tests.torch_tests.kinematic_model import Kinematic,load_kinematic
import numpy as np
import torch
def main() -> None:

    
    # sim_t1,sim_t2 =load_sim_data_dataset()
    # x,y= test_torch_model_sim_data.to_kinematic_data(sim_t1,sim_t2)

    # print('linear ve')
    # print(x.max(axis=0))
    # print(x.min(axis=0))

    # test_torch_model_sim_data_rotate.main()
    # test_torch_model_sim_data.main()

    model = load_kinematic()

    # tensors =[param.data for param in model.parameters()]

    gamma = model.alpha + model.beta

    sin_gamma = gamma.sin()/model.r
    cos_gamma = gamma.cos()/model.r
    l_pred = -model.l * model.beta.cos()/model.r

    weights = torch.cat((sin_gamma.T,-cos_gamma.T,l_pred.T),1)

    print(weights)


   
   

    # model.to_jit_script()

    pass


if __name__ == '__main__':
    main()
