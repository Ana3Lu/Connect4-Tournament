"""
Implementación base de MCTS con UCT (single-agent).

Contiene dos funciones:
  - sample_action_from_inner_stats: convierte estadísticas de simulaciones internas
    en una distribución de probabilidad y muestrea una acción.
  - mcts_uct: árbol de búsqueda Monte-Carlo con selección UCT para un solo agente.
"""

import math
from typing import Any, Callable, Dict, Iterable, Tuple

import numpy as np


def sample_action_from_inner_stats(
    q_local: Dict[Any, float],  # valor promedio por acción en simulaciones
    n_local: Dict[Any, int],    # visitas por acción en simulaciones
    legal_actions: Iterable[Any],
    *,
    use_q: bool = True,
    use_counts: bool = True,
    rng: np.random.RandomState,
    eps: float = 1e-6,          # garantiza que ninguna acción quede con prob cero
) -> Tuple[Any, Dict[Any, float]]:
    """
    Muestrea una acción a partir de estadísticas de los inner trials.

    Construye un score por acción combinando valor estimado y visitas:
        score(a) = (max(q(a), 0) + eps) * (N(a) + 1)

    Normalizar ese score entre la suma total da las probabilidades de muestreo.
    Esto implementa trial-based online policy improvement: la experiencia acumulada
    en simulaciones guía la decisión en el proceso externo.
    """
    actions = list(legal_actions) 
    if not actions:
        raise ValueError("legal_actions is empty")

    # calcula score por acción
    scores = np.array([
        ((max(q_local.get(a, 0.0), 0.0) if use_q else 0.0) + eps)
        * ((n_local.get(a, 0) if use_counts else 0) + 1)  # +1 para acciones no visitadas
        for a in actions
    ])

    probs_arr = scores / scores.sum()  # divide por la suma total para que sumen 1
    chosen = rng.choice(len(actions), p=probs_arr)
    # enumerate asocia cada posición del array de probs a su acción correspondiente
    probs = {a: float(probs_arr[i]) for i, a in enumerate(actions)}
    
    return actions[chosen], probs


def mcts_uct(
    root_state: Any,
    legal_actions_fn: Callable[[Any], Iterable[Any]], # acciones legales para un estado dado
    successor_fn: Callable[[Any, Any, np.random.RandomState], Any],
    terminal_fn: Callable[[Any], bool],
    reward_fn: Callable[[Any], float],
    *,
    num_simulations: int,
    max_depth: int,
    exploration_c: float,
    rng: np.random.RandomState,
) -> Dict[str, Any]:
    """
    MCTS single-agent con selección UCT.

    Mantiene tres estructuras a lo largo de todas las simulaciones:
      N_s[(s)]      — visitas al estado s
      N_sa[(s, a)]  — visitas al par estado-acción
      Q_sa[(s, a)]  — valor estimado promedio de tomar a desde s

    Cada simulación sigue tres fases:
      1. Selección y expansión: recorre el árbol con UCT hasta un nodo hoja
      2. Rollout: política aleatoria hasta terminal o max_depth
      3. Backpropagación: actualiza Q con media incremental
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

            # prioriza acciones no visitadas, luego aplica UCT sobre las visitadas
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

            # agrega nodo al camino recorrido para backpropagación posterior
            path.append((s, a))
            N_s[s]       = N_s.get(s, 0) + 1
            N_sa[(s, a)] = N_sa.get((s, a), 0) + 1
            Q_sa.setdefault((s, a), 0.0)
            s = successor_fn(s, a, rng)
            depth += 1

        # rollout aleatorio desde el nodo expandido
        while not terminal_fn(s) and depth < max_depth:
            actions = list(legal_actions_fn(s))
            if not actions:
                break
            a = actions[rng.randint(len(actions))] # política aleatoria simple
            s = successor_fn(s, a, rng)
            depth += 1

        # backpropagación con media incremental sobre el camino recorrido
        R = reward_fn(s) if terminal_fn(s) else 0.0
        for s_p, a_p in path:
            Q_sa[(s_p, a_p)] += (R - Q_sa[(s_p, a_p)]) / N_sa[(s_p, a_p)]

    # estadísticas de la raíz para decidir qué acción recomendar
    root_actions = list(legal_actions_fn(root_state))
    q_root = {a: Q_sa[(root_state, a)] for a in root_actions if (root_state, a) in N_sa}
    n_root = {a: N_sa[(root_state, a)] for a in root_actions if (root_state, a) in N_sa}

    # acción con mejor estimación (desempate determinista con sorted)
    best_action = max(sorted(q_root.keys()), key=lambda a: q_root[a]) if q_root else None

    return {"q_root": q_root, "n_root": n_root, "best_action": best_action}
