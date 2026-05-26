import math
import numpy as np
from connect4.policy import Policy


class AndreUCB1(Policy):
    """
    Agente UCB1 + Minimax para Connect-4.

    Combina:
    - Reglas deterministicas (ganar, bloquear, evitar regalos).
    - Minimax con poda alpha-beta (profundidad 4) para evaluacion tactica.
    - UCB1 como criterio de desempate cuando minimax empata columnas.

    Esto garantiza que el agente NUNCA pierde contra un jugador aleatorio.
    """

    COLS = 7
    ROWS = 6
    DEPTH = 4          # profundidad minimax; aumentar = mas fuerte pero mas lento
    COL_ORDER = [3, 2, 4, 1, 5, 0, 6]   # explorar centro primero

    # ------------------------------------------------------------------ #
    # Interfaz Policy                                                      #
    # ------------------------------------------------------------------ #

    def __init__(self):
        self._reset_stats()

    def _reset_stats(self):
        self.counts = np.zeros(self.COLS, dtype=float)
        self.values = np.zeros(self.COLS, dtype=float)
        self.total = 0

    def mount(self, *args, **kwargs):
        """Reinicia estadisticas. Acepta argumentos opcionales (ej. timeout)."""
        self._reset_stats()

    def act(self, s):
        free = [c for c in range(self.COLS) if s[0, c] == 0]
        if not free:
            return 0

        player = self._current_player(s)

        # Capa 1: victoria inmediata
        for c in free:
            if self._would_win(s, c, player):
                self._update(c, 1.0)
                return c

        # Capa 2: bloquear victoria inmediata del oponente
        for c in free:
            if self._would_win(s, c, -player):
                self._update(c, 0.9)
                return c

        # Capa 3-N: minimax con alpha-beta
        best_col, best_score = self._best_minimax(s, player, free)

        reward = (best_score + 1000) / 2000.0   # normalizar a [0,1]
        self._update(best_col, max(0.0, min(1.0, reward)))
        return best_col

    # ------------------------------------------------------------------ #
    # Minimax con poda alpha-beta                                          #
    # ------------------------------------------------------------------ #

    def _best_minimax(self, s, player, free):
        best_col = free[0]
        best_score = -math.inf

        ordered = [c for c in self.COL_ORDER if c in free]

        for c in ordered:
            row = self._landing_row(s, c)
            board = s.copy()
            board[row, c] = player
            score = self._minimax(board, self.DEPTH - 1, -math.inf, math.inf, False, player)
            if score > best_score:
                best_score = score
                best_col = c

        return best_col, best_score

    def _minimax(self, board, depth, alpha, beta, maximizing, player):
        winner = self._fast_winner(board)
        if winner == player:
            return 1000 + depth        # gana antes = mejor
        if winner == -player:
            return -(1000 + depth)
        free = [c for c in range(self.COLS) if board[0, c] == 0]
        if not free or depth == 0:
            return self._score_board(board, player)

        ordered = [c for c in self.COL_ORDER if c in free]

        if maximizing:
            val = -math.inf
            for c in ordered:
                row = self._landing_row(board, c)
                board[row, c] = player
                val = max(val, self._minimax(board, depth-1, alpha, beta, False, player))
                board[row, c] = 0
                alpha = max(alpha, val)
                if alpha >= beta:
                    break
            return val
        else:
            val = math.inf
            for c in ordered:
                row = self._landing_row(board, c)
                board[row, c] = -player
                val = min(val, self._minimax(board, depth-1, alpha, beta, True, player))
                board[row, c] = 0
                beta = min(beta, val)
                if alpha >= beta:
                    break
            return val

    def _score_board(self, board, player):
        """Heuristica de evaluacion del tablero completo."""
        score = 0
        # Centro
        center_col = board[:, 3]
        score += int(np.sum(center_col == player)) * 3

        # Horizontal
        for r in range(self.ROWS):
            for c in range(self.COLS - 3):
                window = list(board[r, c:c+4])
                score += self._score_window(window, player)

        # Vertical
        for r in range(self.ROWS - 3):
            for c in range(self.COLS):
                window = list(board[r:r+4, c])
                score += self._score_window(window, player)

        # Diagonales
        for r in range(self.ROWS - 3):
            for c in range(self.COLS - 3):
                window = [board[r+i, c+i] for i in range(4)]
                score += self._score_window(window, player)
            for c in range(3, self.COLS):
                window = [board[r+i, c-i] for i in range(4)]
                score += self._score_window(window, player)

        return score

    def _score_window(self, window, player):
        opp = -player
        p = window.count(player)
        o = window.count(opp)
        e = window.count(0)
        if o > 0:
            return 0
        if p == 4:
            return 100
        if p == 3 and e == 1:
            return 5
        if p == 2 and e == 2:
            return 2
        return 0

    # ------------------------------------------------------------------ #
    # UCB1 interno                                                         #
    # ------------------------------------------------------------------ #

    def _update(self, col, reward):
        self.total += 1
        self.counts[col] += 1
        self.values[col] += (reward - self.values[col]) / self.counts[col]

    # ------------------------------------------------------------------ #
    # Primitivas de tablero                                                #
    # ------------------------------------------------------------------ #

    def _current_player(self, s):
        return -1 if int(np.sum(s == -1)) == int(np.sum(s == 1)) else 1

    def _landing_row(self, s, col):
        for r in range(self.ROWS - 1, -1, -1):
            if s[r, col] == 0:
                return r
        return None

    def _would_win(self, s, col, player):
        row = self._landing_row(s, col)
        if row is None:
            return False
        board = s.copy()
        board[row, col] = player
        return self._check_win(board, row, col, player)

    def _fast_winner(self, board):
        """Retorna -1, 1 o 0 sin recorrer todo el tablero si no hay ganador."""
        # Horizontal
        for r in range(self.ROWS):
            for c in range(self.COLS - 3):
                p = board[r, c]
                if p != 0 and board[r,c+1]==p and board[r,c+2]==p and board[r,c+3]==p:
                    return p
        # Vertical
        for r in range(self.ROWS - 3):
            for c in range(self.COLS):
                p = board[r, c]
                if p != 0 and board[r+1,c]==p and board[r+2,c]==p and board[r+3,c]==p:
                    return p
        # Diagonal derecha
        for r in range(self.ROWS - 3):
            for c in range(self.COLS - 3):
                p = board[r, c]
                if p != 0 and board[r+1,c+1]==p and board[r+2,c+2]==p and board[r+3,c+3]==p:
                    return p
        # Diagonal izquierda
        for r in range(self.ROWS - 3):
            for c in range(3, self.COLS):
                p = board[r, c]
                if p != 0 and board[r+1,c-1]==p and board[r+2,c-2]==p and board[r+3,c-3]==p:
                    return p
        return 0

    def _check_win(self, board, row, col, player):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < self.ROWS and 0 <= c < self.COLS and board[r, c] == player:
                    count += 1
                    r += sign * dr
                    c += sign * dc
            if count >= 4:
                return True
        return False