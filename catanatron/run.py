import random

from catanatron.game import Game
from catanatron.models.player import Player, RandomPlayer, Color
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron.players.mcts import MCTSPlayer
from catanatron.players.playouts import GreedyPlayoutsPlayer

class FirstPlayer(Player):
    def decide(self, game, playable_actions):
        """
        Args:
            game (Game): complete game state. read-only.
            playable_actions (Iterable[Action]): options to choose from
        Return:
            action (Action): Chosen element of playable_actions
        """
        return random.choice(playable_actions)
    
players = [
    AlphaBetaPlayer(Color.RED, prunning=True),
    WeightedRandomPlayer(Color.BLUE),
]
game = Game(players)

from pprint import pprint
from catanatron.cli.play import play_batch

wins, results_by_player, games = play_batch(100, players)