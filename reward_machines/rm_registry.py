from reward_machines.games.asteroids_rm import AsteroidsRm
from reward_machines.games.enduro_rm import EnduroRm
from reward_machines.games.pong_rm import PongRm
from reward_machines.games.frostbite_rm import FrostbiteRm
from reward_machines.games.freeway_rm import FreewayRm
from reward_machines.games.seaquest_rm import SeaquestRm
from reward_machines.games.phoenix_rm import PhoenixRm
from reward_machines.games.tennis_rm import TennisRm
from reward_machines.games.kangaroo_rm import KangarooRm
from reward_machines.games.beamrider_rm import BeamriderRm
from reward_machines.games.mspacman_rm import MsPacmanRm
from reward_machines.games.breakout_rm import BreakoutRm
from reward_machines.games.gravitar_rm import GravitarRm
from reward_machines.games.montezumarevenge_rm import MontezumaRm
from reward_machines.games.skiing_rm import SkiingRm

GAME_RM_REGISTRY = {
    "pong": PongRm,
    "seaquest": SeaquestRm,
    "frostbite": FrostbiteRm,
    "freeway": FreewayRm,
    "phoenix": PhoenixRm,
    "tennis": TennisRm,
    "kangaroo": KangarooRm,
    "beamrider": BeamriderRm,
    "phoenix": PhoenixRm,
    "enduro": EnduroRm,
    "asteroids": AsteroidsRm,
    "mspacman": MsPacmanRm,
    "breakout": BreakoutRm,
    "gravitar": GravitarRm,
    "montezumarevenge": MontezumaRm,
    "skiing": SkiingRm,
}
