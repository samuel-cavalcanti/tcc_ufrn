import numpy as np
from pid import PID


class FredericoController:
    __position_controller: PID
    __orientation_controller: PID

    def __init__(self, position_controller: PID, orientation_controller: PID):
        """
            Esse Controlador buscar chegar no ponto de referência independente da sua orientação, para isso ele utiliza
            um PID que envia sinal de velocidade linear e um PID que envia sinal de velocidade angular, o PID que envia
            sinal de velocidade linear busca minimizar a distância da posição do robô para a distância próxima desejada.
            O PID que envia sinais de velocidade angular busca modificar a orientação do robô para que essa distância
            próxima seja a distância real entre o robô o seu alvo.
         """
        self.__position_controller = position_controller
        self.__orientation_controller = orientation_controller

    def step(self, current: np.ndarray, desired_pos: np.ndarray) -> np.ndarray:
        """Calculo da referencia"""
        delta_x = desired_pos[0] - current[0]
        delta_y = desired_pos[1] - current[1]

        delta_l = np.sqrt(delta_x ** 2 + delta_y ** 2)
        phi = np.arctan2(delta_y, delta_x)
        delta_phi = phi - current[2]

        """Controladores desacoplados"""
        linear_velocity_in_polar_system = self.__position_controller.step(
            delta_l * np.cos(delta_phi))
        angular_velocity = self.__orientation_controller.step(delta_phi)

        """Transformando o sistema polar para cartesiano"""
        x = linear_velocity_in_polar_system * np.cos(current[2])
        y = linear_velocity_in_polar_system * np.sin(current[2])

        return np.array([x, y, angular_velocity])
