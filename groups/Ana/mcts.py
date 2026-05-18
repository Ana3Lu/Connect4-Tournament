"""
MCTS con UCT adaptado a Connect-4.

Cambios respecto a la versión base:
  - successor_fn(s, a) sin rng porque ConnectState.transition es determinista
  - reward_fn recibe root_player para evaluar el resultado desde su perspectiva
  - la función se renombra a mcts_uct_two_player anticipando la adaptación two-player

Limitación de esta versión: la backpropagación todavía propaga R igual en todos
los niveles, como en single-agent. No considera que el oponente juega en contra.
Eso se corrige en el siguiente commit.
"""

import math
from typing import Any, Callable, Dict, Iterable, Tuple

import numpy as np


def mcts_uct_two_player(
    root_state: Any,
    legal_actions_fn: Callable[[Any], Iterable[Any]],
    successor_fn: Callable[[Any, Any], Any],           # sin rng, transición determinista
    terminal_fn: Callable[[Any], bool],
    reward_fn: Callable[[Any, int], float],            # recibe root_player para perspectiva
    root_player: int,
    *,
    num_simulations: int,
    max_depth: int,
    exploration_c: float,
    rng: np.random.RandomState,
) -> Dict[str, Any]:
    """
    MCTS con UCT para Connect-4, partiendo de la implementación single-agent base.

    Mantiene las mismas tres estructuras:
      N_s[(s)]      — visitas al estado s
      N_sa[(s, a)]  — visitas al par estado-acción
      Q_sa[(s, a)]  — valor estimado promedio de tomar a desde s

    Cada simulación sigue las mismas tres fases:
      1. Selección y expansión: recorre el árbol con UCT hasta un nodo hoja
      2. Rollout: política aleatoria hasta terminal o max_depth
      3. Backpropagación: actualiza Q con media incremental

    Nota: la backprop aún no niega la recompensa por nivel, lo que sería necesario
    para modelar correctamente que el oponente juega en contra. Próximo commit.
    """
    N_s:  Dict[Any, int]               = {}
    N_sa: Dict[Tuple[Any, Any], int]   = {}
    Q_sa: Dict[Tuple[Any, Any], float] = {}

    for _ in range(num_simulations):
        path = []  # pares (s, a) visitados, necesarios para backprop
        s = root_state
        depth = 0

        # selección y expansión
        while not terminal_fn(s) and depth < max_depth:
            actions = list(legal_actions_fn(s))
            if not actions:
                break

            unvisited = [a for a in actions if (s, a) not in N_sa]
            if unvisited:
                a = unvisited[0]  # orden determinista al expandir nodos nuevos
            else:
                ns = N_s.get(s, 0)
                # key=lambda indica a max() con qué valor comparar cada acción
                a = max(
                    actions,
                    key=lambda a: Q_sa[(s, a)] + exploration_c * math.sqrt(
                        math.log(ns + 1) / (N_sa[(s, a)] + 1)
                    ),
                )

            path.append((s, a))
            N_s[s]       = N_s.get(s, 0) + 1
            N_sa[(s, a)] = N_sa.get((s, a), 0) + 1
            Q_sa.setdefault((s, a), 0.0)
            s = successor_fn(s, a)  # sin rng, ConnectState.transition es determinista
            depth += 1

        # rollout aleatorio desde el nodo expandido
        while not terminal_fn(s) and depth < max_depth:
            actions = list(legal_actions_fn(s))
            if not actions:
                break
            a = actions[rng.randint(len(actions))]
            s = successor_fn(s, a)
            depth += 1

        # backpropagación con media incremental sobre el camino recorrido
        R = reward_fn(s, root_player) if terminal_fn(s) else 0.0
        for s_p, a_p in path:
            Q_sa[(s_p, a_p)] += (R - Q_sa[(s_p, a_p)]) / N_sa[(s_p, a_p)]

    # estadísticas de la raíz para decidir qué acción recomendar
    root_actions = list(legal_actions_fn(root_state))
    q_root = {a: Q_sa[(root_state, a)] for a in root_actions if (root_state, a) in N_sa}
    n_root = {a: N_sa[(root_state, a)] for a in root_actions if (root_state, a) in N_sa}

    best_action = max(sorted(q_root.keys()), key=lambda a: q_root[a]) if q_root else None  # sorted para desempate determinista

    return {"q_root": q_root, "n_root": n_root, "best_action": best_action}
