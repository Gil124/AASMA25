import random
from pprint import pprint
from catanatron.cli.play import play_batch

from catanatron.game import Game
from catanatron.models.player import Player, RandomPlayer, Color
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron.players.mcts import MCTSPlayer
from catanatron.players.playouts import GreedyPlayoutsPlayer

players = [
    AlphaBetaPlayer(Color.RED, prunning=True),
    AlphaBetaPlayer(Color.WHITE, prunning=False),
]

wins, results_by_player, games = play_batch(100, players)