from catanatron.cli.play import play_batch

from catanatron.models.player import Color
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer

players = [
    AlphaBetaPlayer(Color.RED, prunning=True),
    AlphaBetaPlayer(Color.WHITE, prunning=False),
    WeightedRandomPlayer(Color.BLUE),
    WeightedRandomPlayer(Color.ORANGE),
]

wins, results_by_player, games = play_batch(10, players)