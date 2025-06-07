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
    MCTSPlayer(Color.RED, num_simulations=20),
    WeightedRandomPlayer(Color.BLUE),
]
game = Game(players)

from catanatron.cli.play import play_batch

from catanatron.models.player import Color
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer

players = [
    AlphaBetaPlayer(Color.RED, prunning=True),
    AlphaBetaPlayer(Color.WHITE, prunning_improved=True),
]

wins, results_by_player, games = play_batch(20, players)
