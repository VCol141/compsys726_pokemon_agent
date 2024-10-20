from functools import cached_property
import numpy as np
from pyboy.utils import WindowEvent
from pyboy_environment.environments.pokemon.pokemon_environment import PokemonEnvironment
from PIL import Image
import math as mt
import cv2 as cv

class PokemonBrock(PokemonEnvironment):
    def __init__(self, act_freq: int, emulation_speed: int = 0, headless: bool = False) -> None:
        valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN, WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START
        ]
        
        release_button = [
            WindowEvent.RELEASE_ARROW_DOWN, WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A, WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_BUTTON_START
        ]

        super().__init__(
            act_freq=act_freq,
            task="brock",
            init_name="has_pokedex.state",
            emulation_speed=emulation_speed,
            valid_actions=valid_actions,
            release_button=release_button,
            headless=headless
        )

        self.previos_pos = [0, 0]
        self.dist_from_exit = 99
        self.rooms = []
        self.locations = []
        self.reset_game_stats()

    def reset_game_stats(self):
        self.current_hp = 0
        self.current_xp = 0
        self.current_level = 0
        self.current_badges = 0
        self.current_money = 0
        self.ticks = 0
        self.x = 0
        self.y = 0

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

        state_vector = np.array([levels[0], type_id[0], hp, xp[0], badges, money, x, y, map_id])
        return np.concatenate((state_vector, game_area))

    def _calculate_reward(self, new_state: dict) -> float:
        game_location = self._get_location()
        game_stats = self._generate_game_stats()

        total_score = 0
        current_levels_sum = np.array(game_stats["levels"]).sum()
        current_hp = np.array(game_stats["hp"]["current"]).sum()
        current_xp = np.array(game_stats["xp"]).sum()
        current_badges = game_stats["badges"]
        current_money = game_stats["money"]

        # Score changes in levels, HP, XP, badges, and money
        if current_levels_sum > self.current_level:
            total_score += 0.5
        if current_hp > self.current_hp:
            total_score += 0.5
        if current_xp > self.current_xp:
            total_score += 0.5
        if current_badges > self.current_badges:
            total_score += 1
        if current_money > self.current_money:
            total_score += 0.5

        self.update_game_stats(current_levels_sum, current_hp, current_xp, current_badges, current_money)
        
        # Position-based reward logic
        x, y, map_id = game_location['x'], game_location['y'], game_location['map_id']
        if self.steps % 10 == 0 and (x == self.x or y == self.y):
            total_score -= 2
        self.x, self.y = x, y

        if map_id not in self.rooms:
            self.rooms.append(map_id)
            total_score += 10
        elif [x, y] not in self.locations:
            self.locations.append([x, y])
            total_score += 2

        return total_score

    def update_game_stats(self, level, hp, xp, badges, money):
        self.current_level = level
        self.current_hp = hp
        self.current_xp = xp
        self.current_badges = badges
        self.current_money = money

    def _check_if_done(self, game_stats: dict[str, any]) -> bool:
        return game_stats["badges"] > self.prior_game_stats["badges"]

    def _check_if_truncated(self, game_stats: dict) -> bool:
        if self.steps >= 1000:
            self.reset_game_stats()
            return True
        return False
