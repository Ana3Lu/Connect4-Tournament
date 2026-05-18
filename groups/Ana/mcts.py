"""
Implementación de MCTS con UCT para Connect-4 (two-player).

Contiene dos funciones:
  - sample_action_from_inner_stats: convierte estadísticas de simulaciones internas
    en una distribución de probabilidad y muestrea una acción.
  - mcts_uct_two_player: árbol de búsqueda Monte-Carlo con selección UCT adaptado
    a juegos de dos jugadores con perspectiva alternante.
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


def mcts_uct_two_player(
    root_state: Any,
    legal_actions_fn: Callable[[Any], Iterable[Any]], # acciones legales para un estado dado
    successor_fn: Callable[[Any, Any], Any], 
    terminal_fn: Callable[[Any], bool],
    reward_fn: Callable[[Any, int, bool], float],
    root_player: int, # jugador desde cuya perspectiva se evalúan las recompensas
    *, 
    num_simulations: int,
    max_depth: int,
    exploration_c: float,
    reward_shaping: bool, # activa bonificación intermedia por amenazas de 3 en línea
    rng: np.random.RandomState,
    N_s:  Dict[Any, int] | None = None,            # árbol persistente: visitas por estado
    N_sa: Dict[Tuple[Any, Any], int] | None = None, # árbol persistente: visitas por (s,a)
    Q_sa: Dict[Tuple[Any, Any], float] | None = None, # árbol persistente: valores estimados
) -> Dict[str, Any]:
    """
    MCTS con UCT para Connect-4 con perspectiva two-player.

    Mantiene tres estructuras a lo largo de todas las simulaciones:
      N_s[(s)]      — visitas al estado s
      N_sa[(s, a)]  — visitas al par estado-acción
      Q_sa[(s, a)]  — valor estimado promedio de tomar a desde s

    Cada simulación sigue tres fases:
      1. Selección y expansión: recorre el árbol con UCT hasta un nodo hoja
      2. Rollout: política aleatoria hasta terminal o max_depth
      3. Backpropagación: actualiza Q negando R en niveles del oponente,
         porque lo que es bueno para uno es malo para el otro
    """
    # si se pasan diccionarios externos se reutilizan, si no se crean nuevos cada vez
    N_s  = N_s  if N_s  is not None else {}
    N_sa = N_sa if N_sa is not None else {}
    Q_sa = Q_sa if Q_sa is not None else {}

    for _ in range(num_simulations):
        path = []  # pares (s, a) visitados, necesarios para backprop
        s = root_state
        depth = 0

        # selección y expansión
        while not terminal_fn(s) and depth < max_depth:
            actions = list(legal_actions_fn(s))
            if not actions:
                break

            # prioriza acciones no visitadas para expandir el árbol, luego aplica UCT
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

            # agrega el par (s, a) al camino recorrido para backpropagación posterior
            path.append((s, a))
            N_s[s]       = N_s.get(s, 0) + 1
            N_sa[(s, a)] = N_sa.get((s, a), 0) + 1
            Q_sa.setdefault((s, a), 0.0)
            s = successor_fn(s, a)  # sin rng dado que ConnectState.transition es determinista
            depth += 1

        # rollout aleatorio desde el nodo expandido
        while not terminal_fn(s) and depth < max_depth:
            actions = list(legal_actions_fn(s))
            if not actions:
                break
            # prioriza jugadas que generen victoria inmediata para el jugador actual, si no hay, elige aleatoriamente
            played = False
            for a in actions:
                s_next = successor_fn(s, a)
                if s_next.get_winner() == s.player:
                    s = s_next
                    played = True
                    break
            if not played:
                a = actions[rng.randint(len(actions))] # política aleatoria simple
                s = successor_fn(s, a)
            depth += 1

        # backpropagación: nodos pares son del agente (+R), impares del oponente (-R)
        R = reward_fn(s, root_player, reward_shaping) if terminal_fn(s) else 0.0
        for i, (s_p, a_p) in enumerate(path):
            r = R if i % 2 == 0 else -R
            Q_sa[(s_p, a_p)] += (r - Q_sa[(s_p, a_p)]) / N_sa[(s_p, a_p)]

    # estadísticas de la raíz para decidir qué acción recomendar
    root_actions = list(legal_actions_fn(root_state))
    q_root = {a: Q_sa[(root_state, a)] for a in root_actions if (root_state, a) in N_sa}
    n_root = {a: N_sa[(root_state, a)] for a in root_actions if (root_state, a) in N_sa}

    best_action = max(sorted(q_root.keys()), key=lambda a: q_root[a]) if q_root else None  # sorted para desempate determinista

    return {"q_root": q_root, "n_root": n_root, "best_action": best_action}
