import numpy as np
from connect4.policy import Policy
from connect4.connect_state import ConnectState
try:
    from groups.Ana.mcts import mcts_uct_two_player               # ejecución en gradescope
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

    # calcula bonus por amenazas de 3 en línea
    bonus = _three_in_a_row_bonus(state.board, root_player)

    return float(np.clip(base + 0.1 * bonus, -1.0, 1.0))  # clip para mantener rango [-1, 1]


def _infer_player(s: np.ndarray) -> int:
    # jugador -1 empieza, si hay igual cantidad de fichas le toca a -1
    diff = int(np.sum(s == -1)) - int(np.sum(s == 1))
    return -1 if diff == 0 else 1


def _immediate_move(state: ConnectState, player: int):
    """Retorna columna ganadora o de bloqueo urgente, None si no existe."""
    if state.is_final():
        return None

    free_cols = state.get_free_cols()
    for col in free_cols:
        if state.transition(col).get_winner() == player:   # movimiento ganador
            return col
        
    # comprobar si el oponente tiene una jugada ganadora en su próximo turno y bloquearla 
    for col in free_cols:
        # simula la jugada desde la perspectiva del oponente (player invertido) y si existe una columna que le da la victoria
        opp_state = ConnectState(board=state.board, player=-player)
        if opp_state.transition(col).get_winner() == -player:  # bloqueo urgente
            return col
        
    return None


class BaseAnaPolicy(Policy):
    """
    Clase base reutilizable para todas las versiones del agente.
    """

    reward_shaping = False
    persistent_tree = False
    quick_win = False
    smart_rollout = False

    def __init__(self, num_simulations: int = 200):
        self.num_simulations = num_simulations
        self._rng = np.random.RandomState()  # también en __init__ por si mount no se llama

        self._N_s = {}
        self._N_sa = {}
        self._Q_sa = {}

    def mount(self, timeout=None) -> None:  # timeout en segundos (calibrar simulaciones)
        self._rng = np.random.RandomState()

        if self.persistent_tree:
            self._N_s = {}
            self._N_sa = {}
            self._Q_sa = {}

        if timeout is not None:
            # medido localmente: ~107 sims/seg; factor 0.5 de margen para gradescope
            self.num_simulations = max(50, int(timeout * 107 * 0.5))

    def act(self, s: np.ndarray) -> int:
        root_player = _infer_player(s)
        root = ConnectState(board=s, player=root_player)

        # si el juego ha terminado, jugar una jugada aleatoria entre las columnas libres
        if root.is_final():
            free = root.get_free_cols()
            return int(self._rng.choice(free)) if free else 0

        # si hay jugada ganadora o bloqueo urgente, jugarla sin simular
        if self.quick_win:
            immediate = _immediate_move(root, root_player)
            if immediate is not None:
                return immediate

        # simula con MCTS/UCT desde el estado raíz para encontrar la mejor acción
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
            smart_rollout=self.smart_rollout,
            rng=self._rng,

            N_s=self._N_s if self.persistent_tree else None,
            N_sa=self._N_sa if self.persistent_tree else None,
            Q_sa=self._Q_sa if self.persistent_tree else None,
        )

        best = result["best_action"]
        if best is None:  # no debería pasar, pero por si el árbol no exploró nada
            free = [c for c in range(7) if s[0, c] == 0]
            return int(self._rng.choice(free))
        return int(best)
    
class AnaPolicyV1(BaseAnaPolicy):
    """
    V1 - MCTS con UCT base, rollout aleatorio sin shaping, árbol no persistente.
    """

    pass

class AnaPolicyV2(BaseAnaPolicy):
    """
    V2 - MCTS con UCT con reward shaping por amenazas de 3 en línea y perspectiva two-player
    """
    
    reward_shaping = True

class AnaPolicyFinal(BaseAnaPolicy):
    """
    Versión final con todas las mejoras implementadas:
      - reward shaping por amenazas de 3 en línea
      - árbol persistente entre turnos
      - prioriza jugadas ganadoras o bloqueos urgentes sin simular
      - rollout inteligente que prioriza jugadas que generan victoria inmediata para el jugador actual
    """

    reward_shaping = True
    persistent_tree = True
    quick_win = True
    smart_rollout = True

