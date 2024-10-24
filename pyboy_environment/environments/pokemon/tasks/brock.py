from functools import cached_property

import numpy as np
from pyboy.utils import WindowEvent

from pyboy_environment.environments.pokemon.pokemon_environment import (
    PokemonEnvironment,
)
from pyboy_environment.environments.pokemon import pokemon_constants as pkc
from PIL import Image

import math as mt


import cv2 as cv


class PokemonBrock(PokemonEnvironment):
    def __init__(self, act_freq: int, emulation_speed: int = 0, headless: bool = False,) -> None:

        valid_actions: list[WindowEvent] = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START,
        ]

        release_button: list[WindowEvent] = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_BUTTON_START,
        ]

        super().__init__(
            act_freq=act_freq,
            task="brock",
            init_name="has_pokedex.state",
            emulation_speed=emulation_speed,
            valid_actions=valid_actions,
            release_button=release_button,
            headless=headless,
        )

        self.current_hp = 0
        self.current_xp = 0
        self.current_level = 0
        self.current_badges = 0
        self.current_money = 0
        self.x = 0
        self.y = 0
        self.locations = {}
        self.rooms = []
        self.position_history = []
        self.previous_position = (0, 0)
        self.all_previous_positions = {}
        self.all_previous_maps = []
        self.total_distance = 0
        self.total_scoring = 0
        self.previous_position_from_origin = 0

        self.map_history = []
        self.distance_history = []
        self.score_history = []
        self.grad_history = []

        self.down_presses = 0
        self.left_presses = 0
        self.right_presses = 0
        self.up_presses = 0
        self.a_presses = 0
        self.b_presses = 0
        self.start_presses = 0
        self.select_presses = 0
        self.other_presses = 0

        self.run_span = 10000

        self.reset_game_stats()

    def reset_game_stats(self):
        self.current_hp = 0
        self.current_xp = 0
        self.current_level = 0
        self.current_badges = 0
        self.current_money = 0
        self.x = 0
        self.y = 0
        self.locations.clear()
        self.rooms.clear()
        self.previous_position = (0, 0)
        self.total_distance = 0
        self.total_scoring = 0
        self.position_history.clear()
        self.total_distance = 0
        self.total_scoring = 0
        self.previous_position_from_origin = 0
        self.down_presses = 0
        self.left_presses = 0
        self.right_presses = 0
        self.up_presses = 0
        self.a_presses = 0
        self.b_presses = 0
        self.start_presses = 0
        self.select_presses = 0
        self.other_presses = 0

    def _get_state(self) -> np.ndarray:
        game_stats = self._generate_game_stats()
        game_location = self._get_location()
        game_area = np.array(self.game_area()).ravel()

        levels = np.array(game_stats["levels"])
        type_id = np.array(game_stats["type_id"])
        hp = np.array(game_stats["hp"]["current"]).sum()
        xp = np.array(game_stats["xp"])
        badges = game_stats["badges"]
        money = game_stats["money"]
        x = game_location['x']
        y = game_location['y']
        map_id = game_location['map_id']

        state_vector = np.array(
            [levels[0], type_id[0], hp, xp[0], badges, money, x, y, map_id])
        return np.concatenate((state_vector, game_area))

    def _calculate_reward(self, new_state: dict) -> float:
        game_stats = self._generate_game_stats()

        total_score = 0
        current_levels_sum = np.array(game_stats["levels"]).sum()
        current_hp = np.array(game_stats["hp"]["current"]).sum()
        current_xp = np.array(game_stats["xp"]).sum()
        current_badges = game_stats["badges"]
        current_money = game_stats["money"]

        # Score changes in levels, HP, XP, badges, and money
        if current_levels_sum > self.current_level:
            total_score += (current_levels_sum - self.current_level) * 0.5
        if current_hp > self.current_hp:
            total_score += (current_hp - self.current_hp) * 0.5
        if current_xp > self.current_xp:
            total_score += (current_xp - self.current_xp) * 0.5
        if current_badges > self.current_badges:
            total_score += (current_badges - self.current_badges) * 0.5
        if current_money > self.current_money:
            total_score += (current_money - self.current_money) * 0.5

        self.update_game_stats(
            current_levels_sum, current_hp, current_xp, current_badges, current_money)

        total_score += self._Update_Distance()

        total_score += self.button_update()
        
        self.total_scoring += total_score

        if self.steps == (self.run_span - 1):
            total_score += self.run_evaluation()

        return total_score
    
    def button_update(self):
        if self.current_button == WindowEvent.PRESS_ARROW_DOWN:
            self.down_presses += 1
        elif self.current_button == WindowEvent.PRESS_ARROW_LEFT:
            self.left_presses += 1
        elif self.current_button == WindowEvent.PRESS_ARROW_RIGHT:
            self.right_presses += 1
        elif self.current_button == WindowEvent.PRESS_ARROW_UP:
            self.up_presses += 1
        elif self.current_button == WindowEvent.PRESS_BUTTON_A:
            self.a_presses += 1
        elif self.current_button == WindowEvent.PRESS_BUTTON_B:
            self.b_presses += 1
        elif self.current_button == WindowEvent.PRESS_BUTTON_START:
            self.start_presses += 1
        elif self.current_button == WindowEvent.PRESS_BUTTON_SELECT:
            self.select_presses += 1
        else:  
            self.other_presses += 1
        
        if self.current_button not in self.valid_actions:
            return -50
        
        return 0


    def run_evaluation(self):

        game_location = self._get_location()
        x, y, map_id = game_location['x'], game_location['y'], game_location['map_id']

        total_distance = self.total_distance
        total_score = self.total_scoring

        grad = total_score / total_distance

        grad_score = (grad - (sum(self.grad_history) / len(self.grad_history))
                      ) * 100 if len(self.grad_history) > 0 else 0
        map_score = (len(self.rooms) - (sum(self.map_history) / len(self.map_history))
                     ) * 200 if len(self.map_history) > 0 else 0
        distance_score = (total_distance - (sum(self.distance_history) / len(
            self.distance_history))) * 0.5 if len(self.distance_history) > 0 else 0
        score_score = (total_score - (sum(self.score_history) / len(self.score_history))
                       ) * 10 if len(self.score_history) > 0 else 0

        self.grad_history.append(grad)
        self.map_history.append(len(self.rooms))
        self.distance_history.append(total_distance)
        self.score_history.append(total_score)

        print(f"Grad: {grad_score}, Map: {map_score}, Distance: {distance_score}, Score: {score_score}, total_score: {grad_score + map_score + distance_score + score_score}")
        print(f'up button: {self.up_presses}, down button: {self.down_presses}, left button: {self.left_presses}, right button: {self.right_presses}, a button: {self.a_presses}, b button: {self.b_presses}, start button: {self.start_presses}, select button: {self.select_presses}, other button: {self.other_presses}')

        return grad_score + map_score + distance_score + score_score

    def _Update_Distance(self):

        total_score = 0

        game_location = self._get_location()

        # Position-based reward logic
        x, y, map_id = game_location['x'], game_location['y'], game_location['map_id']
        current_position = (x, y)

        if len(self.position_history) >= 2 and current_position == self.position_history[-2]:
            total_score -= 1  # Penalize back-and-forth movements

        self.position_history.append(current_position)
        if len(self.position_history) > 10:  # Limit history size to last 10 positions
            self.position_history.pop(0)

        under_flag = False

        if map_id not in self.rooms:
            self.rooms.append(map_id)
            self.locations[map_id] = [current_position]
        else:
            map_locations = self.locations.get(map_id, [])

            for (x, y) in map_locations:
                distance = mt.sqrt(
                    (x - current_position[0])**2 + (y - current_position[1])**2)

                if distance <= 1.5:
                    under_flag = True
                    break

            if not under_flag:
                self.locations[map_id].append(current_position)
                # print("location found")

        under_flag = False

        if map_id not in self.all_previous_maps:
            self.all_previous_maps.append(map_id)
            self.all_previous_positions[map_id] = [current_position]
            total_score += 10
            # print("New map found")
        else:
            map_locations = self.all_previous_positions.get(map_id, [])

            for (x, y) in map_locations:
                distance = mt.sqrt(
                    (x - current_position[0])**2 + (y - current_position[1])**2)

                if distance <= 3.5:
                    under_flag = True
                    break

            if not under_flag:
                self.all_previous_positions[map_id].append(current_position)
                total_score += 5
        
        current_position_from_origin = mt.sqrt(current_position[0]**2 + current_position[1]**2)

        total_score += (current_position_from_origin - self.previous_position_from_origin) * 5

        self.previous_position_from_origin = current_position_from_origin

        self.total_distance += mt.sqrt((current_position[0] - self.previous_position[0])**2 + (
            current_position[1] - self.previous_position[1])**2)
        

        self.previous_position = current_position

        return total_score

    def update_game_stats(self, level, hp, xp, badges, money):
        self.current_level = level
        self.current_hp = hp
        self.current_xp = xp
        self.current_badges = badges
        self.current_money = money

    def _check_if_done(self, game_stats: dict[str, any]) -> bool:
        # Setting done to true if agent beats first gym (temporary)
        # if self.total_steps_done >= (20000 * 20):
        #     return True
        return game_stats["badges"] > self.prior_game_stats["badges"]

    def _check_if_truncated(self, game_stats: dict) -> bool:
        # Implement your truncation check logic here

        # Maybe if we run out of pokeballs...? or a max step count
        if self.steps >= self.run_span:
            self.reset_game_stats()

            return True

        return False
