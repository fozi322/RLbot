import numpy as np
import torch
import random
import math

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlbot.utils.structures.quick_chats import QuickChats
from rlgym_compat import GameState

from .agent import Agent
from .nexto_obs import NextoObsBuilder, BOOST_LOCATIONS
from .immortal.agent import Agent as ImmortalAgent
from .immortal.obs.advanced_obs import ExpandAdvancedObs
from .immortal.action.actionparser import ImmortalAction

KICKOFF_CONTROLS = (
        11 * 4 * [SimpleControllerState(throttle=1, boost=True)]
        + 4 * 4 * [SimpleControllerState(throttle=1, boost=True, steer=-1)]
        + 2 * 4 * [SimpleControllerState(throttle=1, jump=True, boost=True)]
        + 1 * 4 * [SimpleControllerState(throttle=1, boost=True)]
        + 1 * 4 * [SimpleControllerState(throttle=1, yaw=0.8, pitch=-0.7, jump=True, boost=True)]
        + 13 * 4 * [SimpleControllerState(throttle=1, pitch=1, boost=True)]
        + 10 * 4 * [SimpleControllerState(throttle=1, roll=1, pitch=0.5)]
)

KICKOFF_NUMPY = np.array([
    [scs.throttle, scs.steer, scs.pitch, scs.yaw, scs.roll, scs.jump, scs.boost, scs.handbrake]
    for scs in KICKOFF_CONTROLS
])


class Nexto(BaseAgent):

    def __init__(self,
                 name,
                 team,
                 index,
                 beta=1,
                 render=False,
                 hardcoded_kickoffs=True,
                 stochastic_kickoffs=True):
        super().__init__(name, team, index)
        self.immortal_obs = ExpandAdvancedObs()
        self.act_parser = ImmortalAction()
        self.immortal_agent = ImmortalAgent()
        self.obs_builder = None
        self.agent = Agent()
        self.tick_skip = 8

        # Beta controls randomness:
        # 1=best action, 0.5=sampling from probability, 0=random, -1=worst action, or anywhere inbetween
        self.beta = beta
        self.render = render
        self.hardcoded_kickoffs = hardcoded_kickoffs
        self.stochastic_kickoffs = stochastic_kickoffs

        self.game_state: GameState = None
        self.controls = None
        self.action = None
        self.immortal_action = None
        self.update_action = True
        self.ticks = 0
        self.prev_time = 0
        self.kickoff_index = -1
        self.field_info = None
        # self.gamemode = None

        # toxic handling
        self.isToxic = False
        self.orangeGoals = 0
        self.blueGoals = 0
        self.demoedCount = 0
        self.lastFrameBall = None
        self.lastFrameDemod = False
        self.demoCount = 0
        self.pesterCount = 0
        self.demoedTickCount = 0
        self.demoCalloutCount = 0
        self.lastPacket = None

    def initialize_agent(self, field_info):
        self.field_info = field_info
        self.obs_builder = NextoObsBuilder(field_info=self.field_info)
        self.game_state = GameState(self.field_info)
        self.ticks = self.tick_skip  # So we take an action the first tick
        self.prev_time = 0
        self.controls = SimpleControllerState()
        self.action = np.zeros(8)
        self.update_action = True
        self.kickoff_index = -1

    def render_attention_weights(self, weights, positions, n=3):
        if weights is None:
            return
        mean_weights = torch.mean(torch.stack(weights), dim=0).numpy()[0][0]

        top = sorted(range(len(mean_weights)),
                     key=lambda i: mean_weights[i],
                     reverse=True)
        top.remove(0)  # Self

        self.renderer.begin_rendering('attention_weights')

        invert = np.array([-1, -1, 1]) if self.team == 1 else np.ones(3)
        loc = positions[0] * invert
        mx = mean_weights[~(np.arange(len(mean_weights)) == 1)].max()
        c = 1
        for i in top[:n]:
            weight = mean_weights[i] / mx
            dest = positions[i] * invert
            color = self.renderer.create_color(255, round(255 * (1 - weight)),
                                               round(255),
                                               round(255 * (1 - weight)))
            self.renderer.draw_string_3d(dest, 2, 2, str(c), color)
            c += 1
            self.renderer.draw_line_3d(loc, dest, color)
        self.renderer.end_rendering()

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        cur_time = packet.game_info.seconds_elapsed
        delta = cur_time - self.prev_time
        self.prev_time = cur_time

        ticks_elapsed = round(delta * 120)
        self.ticks += ticks_elapsed
        self.game_state.decode(packet, ticks_elapsed)

        if self.isToxic:
            self.toxicity(packet)

        if self.update_action and len(self.game_state.players) > self.index:
            self.update_action = False

            player = self.game_state.players[self.index]
            teammates = [
                p for p in self.game_state.players
                if p.team_num == self.team and p != player
            ]
            opponents = [
                p for p in self.game_state.players if p.team_num != self.team
            ]

            self.game_state.players = [player] + teammates + opponents

            obs = self.obs_builder.build_obs(player, self.game_state,
                                             self.action)

            beta = self.beta
            if packet.game_info.is_match_ended:
                # or not (packet.game_info.is_kickoff_pause or packet.game_info.is_round_active): Removed due to kickoff
                beta = 0  # Celebrate with random actions
            if self.stochastic_kickoffs and packet.game_info.is_kickoff_pause:
                beta = 0.5

            car_x_location = packet.game_cars[self.index].physics.location.x
            car_z_location = packet.game_cars[self.index].physics.location.z
            ball_x_location = packet.game_ball.physics.location.x
            ball_z_location = packet.game_ball.physics.location.z
            boost = packet.game_cars[self.index].boost
            RIGHT_SIDE_WALL_DISTANCE_X = 4096
            LEFT_SIDE_WALL_DISTANCE_X = -4096
            # if index is 0 then we're blue, we check the left wall, else we're orange and we check the right
            WALL_LOCATION = LEFT_SIDE_WALL_DISTANCE_X if self.team == 0 else RIGHT_SIDE_WALL_DISTANCE_X
            FLOOR = 0
            distance_between_ball = abs(car_x_location - ball_x_location)
            # we dont check the left wall because immortal can only air dribble from right
            # hence we use a large number for the left wall (2000)
            distance_between_wall = abs(WALL_LOCATION - car_x_location)
            car_distance_between_floor = abs(FLOOR - car_z_location)
            # distance_between_opp = abs(packet.game_cars[1 - self.team].physics.location.x - car_x_location)
            # distance_between_opp = 5000 # just for testing
            ball_distance_between_floor = abs(FLOOR - ball_z_location)
            time_remaining = packet.game_info.game_time_remaining
            our_score = packet.teams[self.team].score
            opp_score = packet.teams[1 - self.team].score

            distances = []
            for opp in opponents:
                index = opp.car_id
                opp_x_location = packet.game_cars[index].physics.location.x
                distances.append(abs(car_x_location - opp_x_location))
            distance_between_opp = min(distances)

            up_by = our_score - opp_score

            # wall_check = distance_between_ball < 200 and distance_between_wall < 1500 and boost >= 45 and distance_between_opp > 200
            # floor_check = car_distance_between_floor >= 300 and ball_distance_between_floor >= 300 and not packet.game_cars[self.index].has_wheel_contact and distance_between_ball < 100
            # win_check = (time_remaining < 45 and up_by > 4) or (time_remaining < 210 and up_by > 8) or up_by > 10
            wall_check = distance_between_ball < 200 and (distance_between_wall < 1500) and boost >= 45 and distance_between_opp > 200
            floor_check = car_distance_between_floor >= 300 and ball_distance_between_floor >= 300 and not packet.game_cars[
                self.index].has_wheel_contact and distance_between_ball < 100
            win_check = (time_remaining < 45
                         and up_by >= 4) or (time_remaining < 210
                                             and up_by >= 8) or up_by >= 10

            if wall_check or floor_check or win_check:
                # reason = "car and ball are close to the wall." if wall_check else "we are in the air." if floor_check else "we are winning by a lot."
                # print("\033[H\033[J")
                # print(f"Using Immortal because {reason}")

                immortal_game_state = self.game_state

                player = immortal_game_state.players[self.index]
                teammates = [
                p for p in immortal_game_state.players
                if p.team_num == self.team
                ]
                opponents = [
                    p for p in immortal_game_state.players if p.team_num != self.team
                ]
                if len(opponents) == 0:
                    # There's no opponent, we assume this model is 1v0
                    immortal_game_state.players = [player]
                else:
                    # Sort by distance to ball
                    teammates.sort(key=lambda p: np.linalg.norm(self.game_state.ball.position - p.car_data.position))
                    opponents.sort(key=lambda p: np.linalg.norm(self.game_state.ball.position - p.car_data.position))

                    # Grab opponent in same "position" relative to it's teammates
                    opponent = opponents[min(teammates.index(player), len(opponents) - 1)]

                    immortal_game_state.players = [player, opponent]
                obs = self.immortal_obs.build_obs(player, immortal_game_state,
                                                  self.action)
                action = self.act_parser.parse_actions(
                    self.immortal_agent.act(obs))
                self.action = action
                # obs = self.immortal_obs.build_obs(player, self.game_state, self.immortal_action)
                # self.immortal_action = self.act_parser.parse_actions(self.immortal_agent.act(obs))
            else:
                # print("\033[H\033[J")
                # print("Using Nexto.")
                self.action, weights = self.agent.act(obs, beta)
                # dont boost if we dont have any
                # if self.action[6] > 0 and boost < 1:
                #     self.action[6] = 0

                if self.render:
                    positions = np.asarray(
                        [p.car_data.position for p in self.game_state.players] +
                        [self.game_state.ball.position] + list(BOOST_LOCATIONS))
                    self.render_attention_weights(weights, positions)

        if self.ticks >= self.tick_skip - 1:
            self.update_controls(self.action)

        if self.ticks >= self.tick_skip:
            self.ticks = 0
            self.update_action = True

        if self.hardcoded_kickoffs:
            self.maybe_do_kickoff(packet, ticks_elapsed)

        return self.controls

    def maybe_do_kickoff(self, packet, ticks_elapsed):
        if packet.game_info.is_kickoff_pause:
            if self.kickoff_index >= 0:
                self.kickoff_index += round(ticks_elapsed)
            elif self.kickoff_index == -1:
                is_kickoff_taker = False
                ball_pos = np.array([
                    packet.game_ball.physics.location.x,
                    packet.game_ball.physics.location.y
                ])
                positions = np.array(
                    [[car.physics.location.x, car.physics.location.y]
                     for car in packet.game_cars[:packet.num_cars]])
                distances = np.linalg.norm(positions - ball_pos, axis=1)
                if abs(distances.min() - distances[self.index]) <= 10:
                    is_kickoff_taker = True
                    indices = np.argsort(distances)
                    for index in indices:
                        if abs(distances[index] - distances[self.index]) <= 10 \
                                and packet.game_cars[index].team == self.team \
                                and index != self.index:
                            if self.team == 0:
                                is_left = positions[index,
                                                    0] < positions[self.index,
                                                                   0]
                            else:
                                is_left = positions[index,
                                                    0] > positions[self.index,
                                                                   0]
                            if not is_left:
                                is_kickoff_taker = False  # Left goes

                self.kickoff_index = 0 if is_kickoff_taker else -2

            if 0 <= self.kickoff_index < len(KICKOFF_NUMPY) \
                    and packet.game_ball.physics.location.y == 0:
                action = KICKOFF_NUMPY[self.kickoff_index]
                self.action = action
                self.update_controls(self.action)
        else:
            self.kickoff_index = -1

    def update_controls(self, action):
        self.controls.throttle = action[0]
        self.controls.steer = action[1]
        self.controls.pitch = action[2]
        self.controls.yaw = action[3]
        self.controls.roll = action[4]
        self.controls.jump = action[5] > 0
        self.controls.boost = action[6] > 0
        self.controls.handbrake = action[7] > 0
        # if self.gamemode == "rumble":
        #     self.controls.use_item = np.random.random() > (
        #         self.tick_skip / 1200)  # On average once every 10 seconds

    def _modify_ball_info_for_heatseeker(self, packet, game_state):
        assert self.field_info.num_goals == 2
        target_goal = self.field_info.goals[self.team].location
        target_goal = np.array([target_goal.x, target_goal.y, target_goal.z])

        ball_pos = game_state.ball.position
        ball_vel = game_state.ball.linear_velocity
        vel_mag = np.linalg.norm(ball_vel)

        current_dir = ball_vel / vel_mag

        goal_dir = target_goal - ball_pos
        goal_dist = np.linalg.norm(goal_dir)
        goal_dir = goal_dir / goal_dist

        self.game_state.ball.linear_velocity = 1.1 * vel_mag * (
            goal_dir +
            (current_dir if packet.game_ball.latest_touch.team != self.team
             else goal_dir)) / 2
        self.game_state.inverted_ball.linear_velocity = self.game_state.ball.linear_velocity * np.array(
            [-1, -1, 1])

    def toxicity(self, packet):
        """
        THE SALT MUST FLOW
        """

        # prep the toxic
        scored = False
        scoredOn = False
        demoed = False
        demo = False
        allyChance = False

        player = packet.game_cars[self.index]

        humanMates = [
            p for p in packet.game_cars[:packet.num_cars]
            if p.team == self.team and p.is_bot is False
        ]
        humanOpps = [
            p for p in packet.game_cars[:packet.num_cars]
            if p.team != self.team and p.is_bot is False
        ]
        goodGoal = [0, -5120] if self.team == 0 else [0, 5120]
        badGoal = [0, 5120] if self.team == 0 else [0, -5120]

        if player.is_demolished and self.demoedTickCount == 0:  # and not self.lastFrameDemod:
            demoed = True
            self.demoedTickCount = 120 * 4

        for p in packet.game_cars:
            if p.is_demolished and p.team != self.team and self.demoCalloutCount == 0:  # player is closest
                demo = True
                self.demoCalloutCount = 120 * 4

        if self.blueGoals != packet.teams[0].score:
            # blue goal!
            self.blueGoals = packet.teams[0].score
            if self.team == 0:
                scored = True
            else:
                scoredOn = True

        if self.orangeGoals != packet.teams[1].score:
            # orange goal
            self.orangeGoals = packet.teams[1].score
            if self.team == 1:
                scored = True
            else:
                scoredOn = True

        self.lastPacket = packet

        # ** NaCl **

        if scored:
            i = random.randint(0, 6)
            if i == 0:
                self.send_quick_chat(QuickChats.CHAT_EVERYONE,
                                     QuickChats.Custom_Toxic_GitGut)
                return
            if i == 1:
                self.send_quick_chat(QuickChats.CHAT_EVERYONE,
                                     QuickChats.Compliments_Thanks)
                return

            for p in humanOpps:

                d = math.sqrt((p.physics.location.x - badGoal[0])**2 +
                              (p.physics.location.y - badGoal[1])**2)
                if d < 2000:
                    self.send_quick_chat(QuickChats.CHAT_EVERYONE,
                                         QuickChats.Compliments_WhatASave)
                    i = random.randint(0, 3)
                    if i == 0:
                        self.send_quick_chat(QuickChats.CHAT_EVERYONE,
                                             QuickChats.Reactions_Wow)
                        self.send_quick_chat(QuickChats.CHAT_EVERYONE,
                                             QuickChats.Compliments_WhatASave)
                    return

            for p in humanOpps:
                d = math.sqrt((p.physics.location.x - badGoal[0])**2 +
                              (p.physics.location.y - badGoal[1])**2)
                if d > 9000:
                    self.send_quick_chat(QuickChats.CHAT_EVERYONE,
                                         QuickChats.Reactions_CloseOne)
                    return

        if scoredOn:
            for p in humanMates:
                d = math.sqrt((p.physics.location.x - goodGoal[0])**2 +
                              (p.physics.location.y - goodGoal[1])**2)
                if d < 3000:
                    i = random.randint(0, 2)
                    if i == 0:
                        self.send_quick_chat(QuickChats.CHAT_EVERYONE,
                                             QuickChats.Compliments_NiceBlock)
                    else:
                        self.send_quick_chat(QuickChats.CHAT_EVERYONE,
                                             QuickChats.Compliments_WhatASave)
                    return

            i = random.randint(0, 3)
            if i == 0:
                self.send_quick_chat(QuickChats.CHAT_EVERYONE,
                                     QuickChats.Custom_Excuses_Lag)
                return
            elif i == 1:
                self.send_quick_chat(QuickChats.CHAT_EVERYONE,
                                     QuickChats.Reactions_Okay)
                return

        if demo:
            i = random.randint(0, 2)
            if i == 0:
                self.send_quick_chat(QuickChats.CHAT_EVERYONE,
                                     QuickChats.Custom_Useful_Bumping)

            elif i == 1:
                self.send_quick_chat(QuickChats.CHAT_EVERYONE,
                                     QuickChats.Apologies_Sorry)

            return

        if demoed:
            self.demoCount += 1
            if self.demoCount >= 5:
                i = random.randint(0, 2)
                if i == 0:
                    self.send_quick_chat(QuickChats.CHAT_EVERYONE,
                                         QuickChats.Reactions_Wow)
                self.send_quick_chat(QuickChats.CHAT_EVERYONE,
                                     QuickChats.Custom_Toxic_DeAlloc)

                return

            if self.demoCount >= 3:
                self.send_quick_chat(QuickChats.CHAT_EVERYONE,
                                     QuickChats.Reactions_Wow)
                self.send_quick_chat(QuickChats.CHAT_EVERYONE,
                                     QuickChats.Reactions_Wow)

            self.send_quick_chat(QuickChats.CHAT_EVERYONE,
                                 QuickChats.Reactions_Okay)
            return

        for p in humanMates:
            onOpponentHalf = False

            if p.team == 1 and p.physics.location.y < 0:
                onOpponentHalf = True
            elif p.team == 0 and p.physics.location.y > 0:
                onOpponentHalf = True

            d = math.sqrt((p.physics.location.x -
                           packet.game_ball.physics.location.x)**2 +
                          (p.physics.location.y -
                           packet.game_ball.physics.location.y)**2)
            if d < 1000 and self.pesterCount == 0 and onOpponentHalf:
                self.send_quick_chat(QuickChats.CHAT_EVERYONE,
                                     QuickChats.Information_TakeTheShot)
                self.pesterCount = 120 * 7  # spam but not too much
                return

        if self.demoCalloutCount > 0:
            self.demoCalloutCount -= 1

        if self.demoedTickCount > 0:
            self.demoedTickCount -= 1

        if self.pesterCount > 0:
            self.pesterCount -= 1
