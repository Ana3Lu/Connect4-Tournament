import numpy as np
import pickle
import os
from connect4.policy import Policy



#  Helpers


def _state_key(board: np.ndarray) -> bytes:
    """Convierte el tablero a una clave hashable compacta."""
    return board.tobytes()


def _get_free_cols(board: np.ndarray) -> list[int]:
    return [c for c in range(7) if board[0, c] == 0]


def _drop(board: np.ndarray, col: int, player: int) -> np.ndarray:
    """Aplica una jugada y retorna el nuevo tablero (sin modificar el original)."""
    new_board = board.copy()
    for r in reversed(range(6)):
        if new_board[r, col] == 0:
            new_board[r, col] = player
            break
    return new_board


def _check_winner(board: np.ndarray) -> int:
    """Retorna -1, 1 (ganador) o 0 (sin ganador aún)."""
    for r in range(6):
        for c in range(7):
            p = board[r, c]
            if p == 0:
                continue
            # Horizontal
            if c + 3 < 7 and all(board[r, c+i] == p for i in range(4)):
                return p
            # Vertical
            if r + 3 < 6 and all(board[r+i, c] == p for i in range(4)):
                return p
            # Diagonal ↘
            if r + 3 < 6 and c + 3 < 7 and all(board[r+i, c+i] == p for i in range(4)):
                return p
            # Diagonal ↙
            if r + 3 < 6 and c - 3 >= 0 and all(board[r+i, c-i] == p for i in range(4)):
                return p
    return 0


def _is_final(board: np.ndarray) -> bool:
    return _check_winner(board) != 0 or len(_get_free_cols(board)) == 0



#  Heurística de desempate (usada en act())

def _score_window(window: list[int], player: int) -> float:
    """Puntúa una ventana de 4 celdas a favor del jugador dado."""
    opp = -player
    score = 0.0
    p_count = window.count(player)
    o_count = window.count(opp)
    e_count = window.count(0)

    if p_count == 4:
        score += 100
    elif p_count == 3 and e_count == 1:
        score += 5
    elif p_count == 2 and e_count == 2:
        score += 2

    if o_count == 3 and e_count == 1:
        score -= 4   # bloquear amenaza rival

    return score


def _heuristic(board: np.ndarray, player: int) -> float:
    """Evalúa el tablero con una heurística simple de ventanas."""
    score = 0.0
    # Centro preferido
    center_col = board[:, 3].tolist()
    score += center_col.count(player) * 3

    # Horizontal
    for r in range(6):
        for c in range(4):
            window = board[r, c:c+4].tolist()
            score += _score_window(window, player)

    # Vertical
    for c in range(7):
        for r in range(3):
            window = board[r:r+4, c].tolist()
            score += _score_window(window, player)

    # Diagonales
    for r in range(3):
        for c in range(4):
            window = [board[r+i, c+i] for i in range(4)]
            score += _score_window(window, player)
    for r in range(3):
        for c in range(3, 7):
            window = [board[r+i, c-i] for i in range(4)]
            score += _score_window(window, player)

    return score



#  Agente Q-learning

class QLearningAgent(Policy):
    """
    Agente de Connect-4 basado en Q-learning tabular.

    Durante mount() entrena offline jugando partidas de self-play.
    Durante act() escoge la acción con mayor Q-valor; si el estado
    no fue visitado, recurre a la heurística como fallback.

    Parámetros configurables
    ------------------------
    episodes      : número de partidas de entrenamiento (↑ = mejor, más lento)
    alpha         : tasa de aprendizaje
    gamma         : factor de descuento
    epsilon_start : exploración inicial (ε-greedy)
    epsilon_end   : exploración mínima al final del entrenamiento
    save_path     : ruta donde guardar/cargar la tabla Q (None = no persiste)
    """

    def __init__(
        self,
        episodes: int = 30_000,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        save_path: str | None = "q_table.pkl",
    ):
        self.episodes = episodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.save_path = save_path

        # Tabla Q:  bytes(board) → {col: Q-valor}
        self.q_table: dict[bytes, dict[int, float]] = {}

    # Acceso a la tabla Q 

    def _get_q(self, key: bytes, col: int) -> float:
        return self.q_table.get(key, {}).get(col, 0.0)

    def _set_q(self, key: bytes, col: int, value: float) -> None:
        if key not in self.q_table:
            self.q_table[key] = {}
        self.q_table[key][col] = value

    def _best_q_col(self, key: bytes, free_cols: list[int]) -> int:
        """Columna con mayor Q-valor entre las libres."""
        return max(free_cols, key=lambda c: self._get_q(key, c))

    #Entrenamiento

    def _epsilon(self, episode: int) -> float:
        """Decaimiento lineal de ε."""
        progress = episode / max(self.episodes - 1, 1)
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def _train_episode(self, rng: np.random.Generator, epsilon: float) -> None:
        """Una partida completa de self-play con Q-learning."""
        board = np.zeros((6, 7), dtype=int)
        player = -1   # -1 = Rojo empieza

        history: list[tuple[bytes, int, int]] = []  # (key, col, player)

        while not _is_final(board):
            free = _get_free_cols(board)
            key = _state_key(board)

            # ε-greedy
            if rng.random() < epsilon:
                col = int(rng.choice(free))
            else:
                col = self._best_q_col(key, free)

            history.append((key, col, player))
            board = _drop(board, col, player)
            player = -player

        # Asignar recompensas hacia atrás
        winner = _check_winner(board)
        for key, col, p in reversed(history):
            if winner == 0:
                reward = 0.0          # empate
            elif winner == p:
                reward = 1.0          # victoria
            else:
                reward = -1.0         # derrota

            old_q = self._get_q(key, col)
            # Q(s,a) ← Q(s,a) + α·(r + γ·max_Q(s') - Q(s,a))
            # En estados terminales max_Q(s')=0
            new_q = old_q + self.alpha * (reward - old_q)
            self._set_q(key, col, new_q)

            # Descuento para las transiciones anteriores
            reward = self.gamma * reward

    def _train(self) -> None:
        rng = np.random.default_rng(42)
        for ep in range(self.episodes):
            eps = self._epsilon(ep)
            self._train_episode(rng, eps)

    # mount()

    def mount(self) -> None:
        """
        Llamado una vez antes de comenzar el torneo.
        Carga tabla Q si existe; si no, entrena desde cero y la guarda.
        """
        if self.save_path and os.path.exists(self.save_path):
            with open(self.save_path, "rb") as f:
                self.q_table = pickle.load(f)
            return

        self._train()

        if self.save_path:
            with open(self.save_path, "wb") as f:
                pickle.dump(self.q_table, f)

    #act() 

    def act(self, s: np.ndarray) -> int:
        """
        Elige la columna a jugar.
        1. Si el estado fue visitado → columna con mayor Q-valor.
        2. Si no fue visitado        → fallback a heurística.
        En ambos casos, siempre bloquea/gana en 1 movimiento si es posible.
        """
        free = _get_free_cols(s)

        # Detectar quién soy: la mayoría de fichas de un color
        # El tablero usa -1 (Rojo) y 1 (Amarillo)
        counts = {-1: int(np.sum(s == -1)), 1: int(np.sum(s == 1))}
        # El jugador actual es quien tiene MENOS fichas (o -1 si es el primero)
        me = -1 if counts[-1] <= counts[1] else 1
        opp = -me

        # Prioridad 1: ganar en este turno 
        for col in free:
            test = _drop(s, col, me)
            if _check_winner(test) == me:
                return col

        # Prioridad 2: bloquear victoria rival 
        for col in free:
            test = _drop(s, col, opp)
            if _check_winner(test) == opp:
                return col

        #Prioridad 3: Q-table 
        key = _state_key(s)
        if key in self.q_table:
            return self._best_q_col(key, free)

        #Prioridad 4: heurística
        return max(free, key=lambda c: _heuristic(_drop(s, c, me), me))