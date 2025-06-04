import random
import time

from catanatron.models.player import Player
from catanatron.models.actions import ActionType


WEIGHTS_BY_ACTION_TYPE = {
    ActionType.BUILD_CITY: 10000,
    ActionType.BUILD_SETTLEMENT: 1000,
    ActionType.BUY_DEVELOPMENT_CARD: 100,
}


class WeightedRandomPlayer(Player):
    """
    Player that decides at random, but skews distribution
    to actions that are likely better (cities > settlements > dev cards).
    """

    def __init__(self, color):
        super().__init__(color)
        self.decision_times = []

    def decide(self, game, playable_actions):
        start = time.time()
        bloated_actions = []
        for action in playable_actions:
            weight = WEIGHTS_BY_ACTION_TYPE.get(action.action_type, 1)
            bloated_actions.extend([action] * weight)
        chosen = random.choice(bloated_actions)
        elapsed = (time.time() - start) * 1000
        self.decision_times.append(elapsed)
        return chosen

    def average_decision_time(self):
        if not self.decision_times:
            return 0.0
        return sum(self.decision_times) / len(self.decision_times)
