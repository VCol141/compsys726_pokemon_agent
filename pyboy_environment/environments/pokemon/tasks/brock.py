from functools import cached_property
import numpy as np
from pyboy.utils import WindowEvent
from pyboy_environment.environments.pokemon.pokemon_environment import PokemonEnvironment
from pyboy_environment.environments.pokemon import pokemon_constants as pkc
from PIL import Image
import math as mt
import cv2 as cv


class PokemonBrock(PokemonEnvironment):
    def __init__(self, act_freq: int, emulation_speed: int = 0, headless: bool = False) -> None:
        valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START,
        ]

        release_button = [
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

        # Game stats and position tracking
        self.current_hp = self.current_xp = self.current_level = 0
        self.current_badges = self.current_money = 0
        self.x = self.y = 0
        self.previous_position = (0, 0)
        self.previous_position_from_origin = (0, 0)
        self.locations = {}
        self.rooms = []
        self.map_history = []
        self.distance_history = []
        self.score_history = []
        self.grad_history = []

        # Action press counters
        self.down_presses = self.left_presses = self.right_presses = 0
        self.up_presses = self.a_presses = self.b_presses = 0
        self.start_presses = self.select_presses = self.other_presses = 0

        # Navigation and scoring
        self.total_distance = self.total_scoring = 0
        self.previous_map_id = 0
        self.map_sequence = [40, 0, 12, 1, 13, 50, 51, 47, 3]
        self.map_index = self.current_map_index = 0
        self.max_d = [0] * len(self.map_sequence)
        self.buffer_position = (0, 0)
        self.run_span = 5000

        self.bad_positions = [[40, 4, 1],
                              [0, 1, 17]]
        self.previous_position_bad = [0] * len(self.map_sequence)
        self.button_presses = [0] * 10
        self.reset_game_stats()

    def reset_game_stats(self):
        # Game stats
        self.current_hp = self.current_xp = self.current_level = 0
        self.current_badges = self.current_money = 0
        self.x = self.y = 0
        self.locations = {}
        self.rooms = []
        self.position_history = []
        self.previous_position = (0, 0)
        self.all_previous_positions = {}
        self.all_previous_maps = []
        self.total_distance = 0
        self.total_scoring = 0
        self.previous_position_from_origin = (0, 0)

        # History tracking
        self.map_history = []
        self.distance_history = []
        self.score_history = []
        self.grad_history = []
        self.button_presses = [0] * 10

        # Action press counters
        self.down_presses = self.left_presses = self.right_presses = 0
        self.up_presses = self.a_presses = self.b_presses = 0
        self.start_presses = self.select_presses = self.other_presses = 0

        # Navigation tracking
        self.previous_map_id = 0
        self.map_index = self.current_map_index = 0

    def _get_state(self) -> np.ndarray:
        game_stats = self._generate_game_stats()
        game_location = self._get_location()

        # Construct the main state vector
        state_vector = np.array([ 
            np.array(game_stats["hp"]["current"]).sum(),
            np.array(game_stats["xp"])[0],
            game_stats["money"],
            game_location['x'],
            game_location['y'],
            game_location['map_id'],
            game_stats["in_battle"]
        ])

        return state_vector


    def _calculate_reward(self, new_state: dict) -> float:
        total_score = self.reward_function(new_state)
        total_score += self._update_distance()
        total_score += self.button_update()

        if self.steps % 100 == 0:
            total_score += self.run_evaluation() * 0.1

        self.total_scoring += total_score
        return total_score

    def button_update(self) -> int:
        button_map = {
            WindowEvent.PRESS_ARROW_DOWN: "down_presses",
            WindowEvent.PRESS_ARROW_LEFT: "left_presses",
            WindowEvent.PRESS_ARROW_RIGHT: "right_presses",
            WindowEvent.PRESS_ARROW_UP: "up_presses",
            WindowEvent.PRESS_BUTTON_A: "a_presses",
            WindowEvent.PRESS_BUTTON_B: "b_presses",
            WindowEvent.PRESS_BUTTON_START: "start_presses",
            WindowEvent.PRESS_BUTTON_SELECT: "select_presses"
        }

        if hasattr(self, "current_button") and self.current_button in button_map:
            setattr(self, button_map[self.current_button], getattr(
                self, button_map[self.current_button]) + 1)
            return 0
        else:
            self.other_presses += 1
            return -50

    def saturate(self, value, min_val, max_val):
        return max(min_val, min(value, max_val))

    def run_evaluation(self) -> float:
        grad_score = self._calculate_score(
            "grad_history", self.total_distance, self.total_scoring)
        map_score = self._calculate_score("map_history", len(self.rooms))
        distance_score = self._calculate_score(
            "distance_history", self.total_distance)
        score_score = self._calculate_score(
            "score_history", self.total_scoring)

        self.total_distance = self.total_scoring = 0
        return 0.1 * (grad_score + distance_score + score_score)

    def _calculate_score(self, history_attr, current_value, score_value=0):
        history = getattr(self, history_attr)
        average = sum(history) / len(history) if history else current_value
        score = self.saturate(
            score_value / current_value if abs(current_value) > 0 else 0, -50, 50)
        history.append(score)
        return score - average

    def reward_function(self, new_state: dict) -> float:
        game_stats = self._generate_game_stats()

        reward_multipliers = {
            "levels": 0.5,
            "hp": 0.5,
            "xp": 0.5,
            "badges": 0.5,
            "money": 0.5,
            "in_battle": [500, 1000]  # First index for battle, second for completion
        }
    
        # Calculate rewards based on 'in_battle' status
        total_score = (
            reward_multipliers["in_battle"][game_stats["in_battle"] - 1] if game_stats["in_battle"] > 0 else 0
        )
    
        # Calculate rewards for other stats
        for stat, mult in reward_multipliers.items():
            if stat == "in_battle":
                continue
        
            # Handle cases where game_stats[stat] might be a dictionary
            stat_value = game_stats[stat]
            if isinstance(stat_value, dict) and "current" in stat_value:
                stat_value = stat_value["current"]
        
            # Calculate the difference and apply multiplier
            stat_difference = np.array(stat_value).sum() - getattr(self, f"current_{stat}", 0)
            total_score += stat_difference * mult

            # Update current game stats
            self.update_game_stats(
                np.array(game_stats["levels"]).sum(),
                np.array(game_stats["hp"]["current"]).sum(),
                np.array(game_stats["xp"]).sum(),
                game_stats["badges"],
                game_stats["money"]
            )
    
        return total_score


    def _update_distance(self) -> float:
        game_location = self._get_location()
        current_position = (game_location['x'], game_location['y'])
        map_id = game_location['map_id']
        total_score = 0
    
        # Penalize if the agent remains in the same position
        if self.steps % 100 == 0 and current_position == self.buffer_position:
            total_score -= 10
        self.buffer_position = current_position

        # Update map index based on map_sequence and reward/penalize forward/backward movement
        for i, seq_map_id in enumerate(self.map_sequence):
            if map_id == seq_map_id:
                self.current_map_index = i
                if i < self.map_index:
                    total_score -= 1000  # Penalty for moving backward in map sequence
                elif i > self.map_index:
                    self.map_index = i
                    total_score += 1000  # Reward for progressing to the next map in sequence
                break

        # Calculate and store max distance from origin for each map
        if map_id != self.previous_map_id:
            self.previous_map_id = map_id
            self.previous_position_from_origin = current_position  # Reset origin for new map

        position_diff = mt.sqrt(
            (current_position[0] - self.previous_position_from_origin[0])**2 +
            (current_position[1] - self.previous_position_from_origin[1])**2
        )

        # Update maximum distance for the current map and add score if it increases
        if position_diff > self.max_d[self.current_map_index]:
            self.max_d[self.current_map_index] = position_diff
            total_score += 100  # Reward for increasing max distance on the current map

        # Update tracking variables
        self.total_distance += mt.sqrt(
            (current_position[0] - self.previous_position[0])**2 +
            (current_position[1] - self.previous_position[1])**2
        )

        for positions in self.bad_positions:
            if map_id == positions[0]:
                position = mt.sqrt((current_position[0] - positions[1])**2 + (current_position[1] - positions[2])**2)
                print
                if position < 2:
                    total_score -= 100

                a = (-self.previous_position_bad[self.current_map_index] + position) * 100 if position < 20 else 0
                b = 0

                if map_id == 40:
                    b += (- self.previous_position[1] + current_position[1])
                elif map_id == 0:
                    b += (- current_position[1] + self.previous_position[1])

                total_score += (a + b)
                self.previous_position_bad[self.current_map_index] = position
                break
        
        self.previous_position = current_position

        return total_score


    def _update_map_index(self, map_id, current_position):
        if map_id != self.previous_map_id:
            self.previous_map_id = map_id
            self.previous_position_from_origin = current_position

        position_diff = mt.sqrt(
            (current_position[0] - self.previous_position_from_origin[0])**2 +
            (current_position[1] - self.previous_position_from_origin[1])**2
        )

        if position_diff > self.max_d[self.current_map_index]:
            self.max_d[self.current_map_index] = position_diff

        self.previous_map_id = map_id

    def update_game_stats(self, level, hp, xp, badges, money):
        self.current_level, self.current_hp, self.current_xp = level, hp, xp
        self.current_badges, self.current_money = badges, money

    def _check_if_done(self, game_stats: dict[str, any]) -> bool:

        return (self.total_steps_done > self.run_span *  2 * 10 * 5) or game_stats["badges"] > self.prior_game_stats["badges"]

    def _check_if_truncated(self, game_stats: dict) -> bool:
        if self.steps >= self.run_span:
            self.reset_game_stats()
            return True
        return False
