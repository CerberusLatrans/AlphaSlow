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


"""recreate network class here to instantiate and use in Alphaslow agent"""
class NeuralNet(nn.Module):
    def __init__(self, n_in, n_mid, n_out):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_out)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        outputs = self.sigmoid(self.fc1(inputs))
        outputs = self.tanh(self.fc2(outputs))
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


        """TURNING PACKET INTO NETWORK FEATURE MATRIX"""
        inputs = torch.Tensor(inputs).float()

        """IN PROGRESS- need to regularize inputs"""

        """PROPAGATING FEATURES THROUGH NETWORK TO GET CONTROLS OUTPUT"""
        model = NeuralNet(len(inputs), int((len(inputs)+8)/2), 8)
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
        self.renderer.draw_string_2d(0, 0, 1, 1, str(inputs), self.renderer.cyan())
        self.renderer.draw_string_2d(0, 20, 1, 1, str(len(inputs)), self.renderer.cyan())
        self.renderer.draw_string_2d(0, 40, 1, 1, str(int((len(inputs)+8)/2)), self.renderer.cyan())
        self.renderer.draw_string_2d(0, 60, 1, 1, str(len(output_vector)), self.renderer.cyan())
        self.renderer.end_rendering()
        """RENDERING TEXT TO DEBUG"""
        return controls
