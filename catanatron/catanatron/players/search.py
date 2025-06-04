import random
import time

from catanatron.state_functions import (
    player_key,
)
from catanatron.models.player import Player
from catanatron.game import Game


class VictoryPointPlayer(Player):
    """
    Player that chooses actions by maximizing Victory Points greedily.
    If multiple actions lead to the same max-points-achievable
    in this turn, selects from them at random.
    """

    def __init__(self, color):
        super().__init__(color)
        self.decision_times = []

    def decide(self, game: Game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        start = time.time()
        best_value = float("-inf")
        best_actions = []
        for action in playable_actions:
            game_copy = game.copy()
            game_copy.execute(action)

            key = player_key(game_copy.state, self.color)
            value = game_copy.state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"]
            if value == best_value:
                best_actions.append(action)
            if value > best_value:
                best_value = value
                best_actions = [action]
        elapsed = (time.time() - start) * 1000
        self.decision_times.append(elapsed)
        return random.choice(best_actions)

    def average_decision_time(self):
        if not self.decision_times:
            return 0.0
        return sum(self.decision_times) / len(self.decision_times)
