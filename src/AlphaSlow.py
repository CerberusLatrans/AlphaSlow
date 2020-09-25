from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.messages.flat.QuickChatSelection import QuickChatSelection
from rlbot.utils.structures.game_data_struct import GameTickPacket

from util.ball_prediction_analysis import find_slice_at_time
from util.boost_pad_tracker import BoostPadTracker
from util.drive import steer_toward_target
from util.sequence import Sequence, ControlStep
from util.vec import Vec3

import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import re

src_path = os.path.abspath("") + "\\" + "src" + "\\"

#with open(src_path + "FINAL_PARAMETERS.txt", "r") as file:
with open(src_path + "FINAL_PARAMETERS.txt", "r") as file:
    parameters = file.read().replace('\n', '')
    param_lists = re.findall("\(\[.+?\]\)", parameters)

    fc1_weight_tensor = torch.nn.Parameter(torch.Tensor(eval(param_lists[0].strip("()"))))
    fc1_bias_tensor = torch.nn.Parameter(torch.Tensor(eval(param_lists[1].strip("()"))))

    fc2_weight_tensor = torch.nn.Parameter(torch.Tensor(eval(param_lists[2].strip("()"))))
    fc2_bias_tensor = torch.nn.Parameter(torch.Tensor(eval(param_lists[3].strip("()"))))

    fc3_weight_tensor = torch.nn.Parameter(torch.Tensor(eval(param_lists[4].strip("()"))))
    fc3_bias_tensor = torch.nn.Parameter(torch.Tensor(eval(param_lists[5].strip("()"))))

    fc4_weight_tensor = torch.nn.Parameter(torch.Tensor(eval(param_lists[6].strip("()"))))
    fc4_bias_tensor = torch.nn.Parameter(torch.Tensor(eval(param_lists[7].strip("()"))))

    fc5_weight_tensor = torch.nn.Parameter(torch.Tensor(eval(param_lists[8].strip("()"))))
    fc5_bias_tensor = torch.nn.Parameter(torch.Tensor(eval(param_lists[9].strip("()"))))

    fc6_weight_tensor = torch.nn.Parameter(torch.Tensor(eval(param_lists[10].strip("()"))))
    fc6_bias_tensor = torch.nn.Parameter(torch.Tensor(eval(param_lists[11].strip("()"))))

    fc7_weight_tensor = torch.nn.Parameter(torch.Tensor(eval(param_lists[12].strip("()"))))
    fc7_bias_tensor = torch.nn.Parameter(torch.Tensor(eval(param_lists[13].strip("()"))))

    fc8_weight_tensor = torch.nn.Parameter(torch.Tensor(eval(param_lists[14].strip("()"))))
    fc8_bias_tensor = torch.nn.Parameter(torch.Tensor(eval(param_lists[15].strip("()"))))

    fc9_weight_tensor = torch.nn.Parameter(torch.Tensor(eval(param_lists[16].strip("()"))))
    fc9_bias_tensor = torch.nn.Parameter(torch.Tensor(eval(param_lists[17].strip("()"))))

"""recreate network class here to instantiate and use in Alphaslow agent"""
class NeuralNet(nn.Module):
    def __init__(self, n_in, n_out):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(n_in, 160, bias = True)
        self.fc2 = nn.Linear(160, 160, bias = True)
        self.fc3 = nn.Linear(160, 80, bias = True)
        self.fc4 = nn.Linear(80, 80, bias = True)
        self.fc5 = nn.Linear(80, 40, bias = True)
        self.fc6 = nn.Linear(40, 40, bias = True)
        self.fc7 = nn.Linear(40, 20, bias = True)
        self.fc8 = nn.Linear(20, 20, bias = True)
        self.fc9 = nn.Linear(20, n_out, bias = True)

        self.sigmoid = nn.Sigmoid()
        self.Tanh = nn.Tanh()
        self.ReLU = nn.ReLU()

        with torch.no_grad():
            self.fc1.weight = fc1_weight_tensor
            self.fc1.bias = fc1_bias_tensor
            self.fc2.weight = fc2_weight_tensor
            self.fc2.bias = fc2_bias_tensor
            self.fc3.weight = fc3_weight_tensor
            self.fc3.bias = fc3_bias_tensor
            self.fc4.weight = fc4_weight_tensor
            self.fc4.bias = fc4_bias_tensor
            self.fc5.weight = fc5_weight_tensor
            self.fc5.bias = fc5_bias_tensor
            self.fc6.weight = fc6_weight_tensor
            self.fc6.bias = fc6_bias_tensor
            self.fc7.weight = fc7_weight_tensor
            self.fc7.bias = fc7_bias_tensor
            self.fc8.weight = fc8_weight_tensor
            self.fc8.bias = fc8_bias_tensor
            self.fc9.weight = fc9_weight_tensor
            self.fc9.bias = fc9_bias_tensor

    def forward(self, inputs):
        outputs = self.ReLU(self.fc1(inputs))
        outputs = self.ReLU(self.fc2(outputs))
        outputs = self.ReLU(self.fc3(outputs))
        outputs = self.ReLU(self.fc4(outputs))
        outputs = self.ReLU(self.fc5(outputs))
        outputs = self.ReLU(self.fc6(outputs))
        outputs = self.ReLU(self.fc7(outputs))
        outputs = self.ReLU(self.fc8(outputs))
        outputs = self.Tanh(self.fc9(outputs))

        return outputs

class AlphaSlow(BaseAgent):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.active_sequence: Sequence = None
        self.boost_pad_tracker = BoostPadTracker()
        self.team = team
        self.index = index

    def initialize_agent(self):
        # Set up information about the boost pads now that the game is active and the info is available
        self.boost_pad_tracker.initialize_boosts(self.get_field_info())

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        """
        This function will be called by the framework many times per second. This is where you can
        see the motion of the ball, etc. and return controls to drive your car.
        """

        # Keep our boost pad info updated with which pads are currently active
        self.boost_pad_tracker.update_boost_status(packet)

        # This is good to keep at the beginning of get_output. It will allow you to continue
        # any sequences that you may have started during a previous call to get_output.
        if self.active_sequence and not self.active_sequence.done:
            controls = self.active_sequence.tick(packet)
            if controls is not None:
                return controls

        """COLLECTING FEATURES"""
        if self.team == 1:
            opponent_team = 0
        elif self.team == 0:
            opponent_team = 1

        #empty list to put the input val in to become input tensor
        inputs = []

        def bool_to_int_val(path, val_name):
            if path == True:
                val_name = 1
            elif path == False:
                val_name = 0
            #return val_name
            inputs.append(val_name)

        def int_to_int_val(path, val_name):
            val_name = path
            #return val_name
            inputs.append(val_name)

        def vector_val(path, list_of_val_names_xyz):
            #list(Vec3(path))[0][0]
            list_of_val_names_xyz[0] = path.x
            inputs.append(list_of_val_names_xyz[0])

            list_of_val_names_xyz[1] = path.y
            inputs.append(list_of_val_names_xyz[1])

            list_of_val_names_xyz[2] = path.z
            inputs.append(list_of_val_names_xyz[2])

        def rotator_val(path, list_of_val_names_pyr):
            list_of_val_names_pyr[0] = path.pitch
            inputs.append(list_of_val_names_pyr[0])

            list_of_val_names_pyr[1] = path.yaw
            inputs.append(list_of_val_names_pyr[1])

            list_of_val_names_pyr[2] = path.roll
            inputs.append(list_of_val_names_pyr[2])

        "game cars"
        own_car = packet.game_cars[self.team]
        opp_car = packet.game_cars[opponent_team]

        vector_val(own_car.physics.location, ['car_location_x', 'car_location_y', 'car_location_z'])
        rotator_val(own_car.physics.rotation, ['car_rotation_pitch', 'car_rotation_yaw', 'car_rotation_roll'])
        vector_val(own_car.physics.velocity, ['car_velocity_x', 'car_velocity_y', 'car_velocity_z'])
        vector_val(own_car.physics.angular_velocity, ['car_angular_velocity_x', 'car_angular_velocity_y', 'car_angular_velocity_z'])
        #bool_to_int_val(own_car.is_demolished, 'demo_state')
        #bool_to_int_val(own_car.has_wheel_contact, 'wheel_contact')
        #bool_to_int_val(own_car.is_super_sonic, 'super_sonic')
        #bool_to_int_val(own_car.jumped, 'jumped')
        #bool_to_int_val(own_car.double_jumped, 'double_jumped')

        own_boost_amount = (own_car.boost)
        inputs.append(own_boost_amount)

        vector_val(opp_car.physics.location, ['car_location_x', 'car_location_y', 'car_location_z'])
        rotator_val(opp_car.physics.rotation, ['car_rotation_pitch', 'car_rotation_yaw', 'car_rotation_roll'])
        vector_val(opp_car.physics.velocity, ['car_velocity_x', 'car_velocity_y', 'car_velocity_z'])
        vector_val(opp_car.physics.angular_velocity, ['car_angular_velocity_x', 'car_angular_velocity_y', 'car_angular_velocity_z'])
        #bool_to_int_val(opp_car.is_demolished, 'demo_state')
        #bool_to_int_val(opp_car.has_wheel_contact, 'wheel_contact')
        #bool_to_int_val(opp_car.is_super_sonic, 'super_sonic')
        #bool_to_int_val(opp_car.jumped, 'jumped')
        #bool_to_int_val(opp_car.double_jumped, 'double_jumped')

        opp_boost_amount = (opp_car.boost)
        inputs.append(opp_boost_amount)

        "game boosts"
        #boost_pad = packet.game_boosts

        #for pad in range(34):
            #bool_to_int_val(boost_pad[pad].is_active, 'pad_' + str(pad+1) + '_bool')
            #int_to_int_val(boost_pad[pad].timer, 'pad_' + str(pad+1) + '_timer')

        "game ball"
        ball = packet.game_ball

        vector_val(ball.physics.location, ['ball_location_x', 'ball_location_y', 'ball_location_z'])
        rotator_val(ball.physics.rotation, ['ball_rotation_pitch', 'ball_rotation_yaw', 'ball_rotation_roll'])
        vector_val(ball.physics.velocity, ['ball_velocity_x', 'ball_velocity_y', 'ball_velocity_z'])
        vector_val(ball.physics.angular_velocity, ['ball_angular_velocity_x', 'ball_angular_velocity_y', 'ball_angular_velocity_z'])
        #vector_val(ball.latest_touch.hit_location, ['ball_hit_location_x', 'ball_hit_location_y', 'ball_hit_location_z'])
        #vector_val(ball.latest_touch.hit_normal, ['ball_hit_normal_x', 'ball_hit_normal_y', 'ball_hit_normal_z'])

        #time_since_last_touch = ball.latest_touch.time_seconds
        #inputs.append(time_since_last_touch)

        #last_touch_team = ball.latest_touch.team
        #inputs.append(last_touch_team)

        "game info"
        info = packet.game_info

        #bool_to_int_val(info.is_overtime, 'is_overtime')
        #bool_to_int_val(info.is_round_active, 'is_round_active')
        #bool_to_int_val(info.is_kickoff_pause, 'is_kickoff')

        #time_elapsed = (info.seconds_elapsed)
        #inputs.append(time_elapsed)

        #time_remaining = (info.game_time_remaining)
        #inputs.append(time_remaining)

        "teams"
        own_team = packet.teams[self.team]
        opp_team = packet.teams[opponent_team]

        #score_diff = own_team.score - opp_team.score
        #inputs.append(score_diff)

        """NORMALIZING INPUTS"""
        def norm(inputs, column, min, max):
            new_value = (inputs[column] - min) /  (max - min)
            inputs[column] = new_value
            return inputs

        norm(inputs, 0, -4096, 4096)
        norm(inputs, 1, -4096, 4096)
        norm(inputs, 2, 0, 2044)
        norm(inputs, 3, -math.pi/2, math.pi/2)
        norm(inputs, 4, -math.pi, math.pi)
        norm(inputs, 5, -math.pi, math.pi)
        norm(inputs, 6, -2300, 2300)
        norm(inputs, 7, -2300, 2300)
        norm(inputs, 8, -2300, 2300)
        norm(inputs, 9, -5500, 5500)
        norm(inputs, 10, -5.5, 5.5)
        norm(inputs, 11, -5.5, 5.5)
        norm(inputs, 12, 0, 100)

        norm(inputs, 13, -4096, 4096)
        norm(inputs, 14, -4096, 4096)
        norm(inputs, 15, 0, 2044)
        norm(inputs, 16, -math.pi/2, math.pi/2)
        norm(inputs, 17, -math.pi, math.pi)
        norm(inputs, 18, -math.pi, math.pi)
        norm(inputs, 19, -2300, 2300)
        norm(inputs, 20, -2300, 2300)
        norm(inputs, 21, -2300, 2300)
        norm(inputs, 22, -5.5, 5.5)
        norm(inputs, 23, -5.5, 5.5)
        norm(inputs, 24, -5.5, 5.5)
        norm(inputs, 25, 0, 100)

        norm(inputs, 26, -4096, 4096)
        norm(inputs, 27, -4096, 4096)
        norm(inputs, 28, 0, 2044)
        norm(inputs, 29, -math.pi/2, math.pi/2)
        norm(inputs, 30, -math.pi, math.pi)
        norm(inputs, 31, -math.pi, math.pi)
        norm(inputs, 32, -6000, 6000)
        norm(inputs, 33, -6000, 6000)
        norm(inputs, 34, -6000, 6000)
        norm(inputs, 35, -6, 6)
        norm(inputs, 36, -6, 6)
        norm(inputs, 37, -6, 6)

        """TURNING PACKET INTO NETWORK FEATURE MATRIX"""
        inputs = torch.Tensor(inputs).float()

        """IN PROGRESS- need to regularize inputs"""

        """PROPAGATING FEATURES THROUGH NETWORK TO GET CONTROLS OUTPUT"""
        model = NeuralNet(len(inputs), 8)
        output_vector = model(inputs)
        output_vector = list(output_vector)

        """SENDING CONTROLS"""
        controls = SimpleControllerState()

        controls.throttle = output_vector[0]
        controls.steer = output_vector[1]
        controls.pitch = output_vector[2]
        controls.yaw = output_vector[3]
        controls.roll = output_vector[4]

        if output_vector[5] >= 0:
            controls.jump = True
        else:
            controls.jump = False
        if output_vector[6] >= 0:
            controls.boost = True
        else:
            controls.boost = False
        if output_vector[7] >= 0:
            controls.handbrake = True
        else:
            controls.handbrake = False
        """RENDERING TEXT TO DEBUG"""
        self.renderer.begin_rendering()
        self.renderer.draw_string_2d(0, 0, 1, 1, str(inputs[:13]), self.renderer.yellow())
        self.renderer.draw_string_2d(0, 80, 1, 1, str(inputs[13:26]), self.renderer.yellow())
        self.renderer.draw_string_2d(0, 160, 1, 1, str(inputs[26:]), self.renderer.yellow())
        self.renderer.draw_string_2d(0, 200, 1, 1, str(len(inputs)), self.renderer.yellow())
        self.renderer.draw_string_2d(0, 220, 1, 1, str(int((len(inputs)+8)/2)), self.renderer.yellow())
        self.renderer.draw_string_2d(0, 240, 1, 1, str(len(output_vector)), self.renderer.yellow())
        self.renderer.draw_string_2d(0, 260, 1, 1, str(src_path), self.renderer.yellow())
        self.renderer.end_rendering()
        """RENDERING TEXT TO DEBUG"""
        return controls
