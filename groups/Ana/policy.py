import numpy as np
from connect4.policy import Policy
from connect4.connect_state import ConnectState
try:
    from mcts import mcts_uct_two_player               # ejecución en gradescope
except ModuleNotFoundError:
    from groups.Ana.mcts import mcts_uct_two_player    # ejecución local desde raíz del proyecto


def _three_in_a_row_bonus(board: np.ndarray, player: int) -> float:
    """
    Diferencia de amenazas de 3 en línea entre player y su oponente.
    Una amenaza es una ventana de 4 celdas con 3 fichas propias y 1 vacía.
    
    Un valor positivo indica que player tiene más amenazas activas que el oponente,
    es decir, más caminos abiertos para completar 4 en línea.
    """
    rows, cols = board.shape

    def count_threats(p: int) -> int:
        count = 0
        for r in range(rows):
            for c in range(cols - 3):              # horizontal
                w = board[r, c:c+4]
                if np.sum(w == p) == 3 and np.sum(w == 0) == 1:
                    count += 1
        for r in range(rows - 3):
            for c in range(cols):                  # vertical
                w = board[r:r+4, c]
                if np.sum(w == p) == 3 and np.sum(w == 0) == 1:
                    count += 1
        for r in range(rows - 3):
            for c in range(cols - 3):              # diagonal
                w = [board[r+i, c+i] for i in range(4)]
                if w.count(p) == 3 and w.count(0) == 1:
                    count += 1
        for r in range(rows - 3):
            for c in range(3, cols):               # diagonal inversa
                w = [board[r+i, c-i] for i in range(4)]
                if w.count(p) == 3 and w.count(0) == 1:
                    count += 1
        return count

    return float(count_threats(player) - count_threats(-player))


def _reward(state: ConnectState, root_player: int, shaping: bool) -> float:
    """
    Recompensa terminal desde la perspectiva de root_player.

    Sin shaping el único +1 llega al final de la partida, lo que con pocas
    simulaciones deja al rollout sin guía sobre qué posiciones son prometedoras.

    Con shaping activo se suma una bonificación por amenazas de 3 en línea
    para que el agente también aprenda a valorar estados intermedios favorables.
    """
    winner = state.get_winner()
    if winner == root_player:
        base = 1.0
    elif winner == -root_player:
        base = -1.0
    else:
        base = 0.0

    if not shaping:
        return base

    bonus = _three_in_a_row_bonus(state.board, root_player)
    return float(np.clip(base + 0.1 * bonus, -1.0, 1.0))  # clip para mantener rango [-1, 1]


def _infer_player(s: np.ndarray) -> int:
    # jugador -1 empieza, si hay igual cantidad de fichas le toca a -1
    diff = int(np.sum(s == -1)) - int(np.sum(s == 1))
    return -1 if diff == 0 else 1


class AnaPolicy(Policy):
    """
    Versión 1 — MCTS/UCT two-player para Connect-4.

    En cada turno construye el árbol desde cero con num_simulations simulaciones
    y lo descarta al terminar. Es la base sobre la que se construye la versión 2.

    Tornillos:
      num_simulations — más simulaciones produce decisiones más informadas
      reward_shaping  — activa bonificación intermedia por amenazas de 3 en línea
    """

    def __init__(self, num_simulations: int = 200, reward_shaping: bool = False):
        self.num_simulations = num_simulations
        self.reward_shaping = reward_shaping
        self._rng = np.random.RandomState()  # también en __init__ por si mount no se llama

    def mount(self, timeout=None) -> None:  # timeout ignorado, requerido por gradescope
        self._rng = np.random.RandomState()

    def act(self, s: np.ndarray) -> int:
        root_player = _infer_player(s)
        root = ConnectState(board=s, player=root_player)

        result = mcts_uct_two_player(
            root_state=root,
            legal_actions_fn=lambda st: st.get_free_cols(),
            successor_fn=lambda st, a: st.transition(a),
            terminal_fn=lambda st: st.is_final(),
            reward_fn=_reward,
            root_player=root_player,
            num_simulations=self.num_simulations,
            max_depth=42,        # máximo de movimientos posibles en Connect-4
            exploration_c=1.41,  # valor estándar UCT (sqrt(2))
            reward_shaping=self.reward_shaping,
            rng=self._rng,
        )

        best = result["best_action"]
        if best is None:  # no debería pasar, pero por si el árbol no exploró nada
            free = [c for c in range(7) if s[0, c] == 0]
            return int(self._rng.choice(free))
        return int(best)


class AnaPolicyPersistent(Policy):
    """
    Versión 2 — MCTS/UCT con árbol persistente entre turnos.

    A diferencia de la versión 1, N_s, N_sa y Q_sa no se descartan al terminar
    cada turno sino que se acumulan a lo largo de toda la partida. 
    
    Así cada turno parte de estadísticas ya exploradas en lugar de cero, lo que implementa
    aprendizaje online: el agente mejora su política con la experiencia acumulada.

    Tornillos:
      num_simulations — simulaciones por turno
      reward_shaping  — activa bonificación intermedia por amenazas de 3 en línea
    """

    def __init__(self, num_simulations: int = 200, reward_shaping: bool = False):
        self.num_simulations = num_simulations
        self.reward_shaping = reward_shaping
        self._rng = np.random.RandomState()  # también en __init__ por si mount no se llama
        self._N_s:  dict = {}  # estadísticas del árbol, se acumulan entre turnos
        self._N_sa: dict = {}
        self._Q_sa: dict = {}

    def mount(self, timeout=None) -> None:  # timeout ignorado, requerido por gradescope
        self._rng = np.random.RandomState()
        # reinicia el árbol al comienzo de cada partida nueva
        self._N_s  = {}
        self._N_sa = {}
        self._Q_sa = {}

    def act(self, s: np.ndarray) -> int:
        root_player = _infer_player(s)
        root = ConnectState(board=s, player=root_player)

        result = mcts_uct_two_player(
            root_state=root,
            legal_actions_fn=lambda st: st.get_free_cols(),
            successor_fn=lambda st, a: st.transition(a),
            terminal_fn=lambda st: st.is_final(),
            reward_fn=_reward,
            root_player=root_player,
            num_simulations=self.num_simulations,
            max_depth=42,        # máximo de movimientos posibles en Connect-4
            exploration_c=1.41,  # valor estándar UCT (sqrt(2))
            reward_shaping=self.reward_shaping,
            rng=self._rng,
            N_s=self._N_s,    # se pasan los diccionarios acumulados
            N_sa=self._N_sa,
            Q_sa=self._Q_sa,
        )

        best = result["best_action"]
        if best is None:  # no debería pasar, pero por si el árbol no exploró nada
            free = [c for c in range(7) if s[0, c] == 0]
            return int(self._rng.choice(free))
        return int(best)
