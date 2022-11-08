
from typing import Any
import numpy as np

Array32 = np.ndarray[Any, np.dtype[np.float32]]

KINEMATIC_TF_PATH = 'models/dense_model_kinematic/weighs.h5'
CONTROLLER_TF_PATH = 'models/dense_model_controller/weighs.h5'
ENVIRONMENT_TF_PATH = 'models/dense_model_environment/weighs.h5'
KINEMATIC_TORCH_PATH = 'models/kinematic.pth'
