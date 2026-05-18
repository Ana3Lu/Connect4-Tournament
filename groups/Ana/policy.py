import numpy as np
from connect4.policy import Policy
from connect4.connect_state import ConnectState
try:
    from mcts import mcts_uct               # ejecución en gradescope
except ModuleNotFoundError:
    from groups.Ana.mcts import mcts_uct    # ejecución local desde raíz del proyecto


def _reward(state: ConnectState, root_player: int) -> float:
    """
    Recompensa terminal desde la perspectiva de root_player.
    +1 si ganó, -1 si perdió, 0 si empate.
    """
    winner = state.get_winner()
    if winner == root_player:
        return 1.0
    elif winner == -root_player:
        return -1.0
    return 0.0


def _infer_player(s: np.ndarray) -> int:
    # jugador -1 empieza, si hay igual cantidad de fichas le toca a -1
    diff = int(np.sum(s == -1)) - int(np.sum(s == 1))
    return -1 if diff == 0 else 1


class AnaPolicy(Policy):
    """
    Agente MCTS/UCT para Connect-4.
    Tornillo: num_simulations — más simulaciones produce decisiones más informadas.
    """

    def __init__(self, num_simulations: int = 200):
        self.num_simulations = num_simulations
        self._rng = None  # se inicializa en mount antes de cada partida

    def mount(self) -> None:
        self._rng = np.random.RandomState()

    def act(self, s: np.ndarray) -> int:
        root_player = _infer_player(s)
        root = ConnectState(board=s, player=root_player)

        result = mcts_uct(
            root_state=root,
            legal_actions_fn=lambda st: st.get_free_cols(),
            successor_fn=lambda st, a, rng: st.transition(a),  # rng ignorado (transición determinista)
            terminal_fn=lambda st: st.is_final(),
            reward_fn=lambda st: _reward(st, root_player),
            num_simulations=self.num_simulations,
            max_depth=42,        # máximo de movimientos posibles en Connect-4
            exploration_c=1.41,  # valor estándar UCT (sqrt(2))
            rng=self._rng,
        )

        best = result["best_action"]
        if best is None:  # no debería pasar, pero por si el árbol no exploró nada
            free = [c for c in range(7) if s[0, c] == 0]
            return int(self._rng.choice(free))
        return int(best)
