from reward_machines.games.pong_rm import PongRm
from reward_machines.games.seaquest import SeaquestRm


GAME_RM_REGISTRY = {
    "pong": PongRm,
    "seaquest": SeaquestRm,
}
