# AnaPolicy — MCTS/UCT para Connect-4

## Descripción

**AnaPolicy** es un agente para Connect-4 basado en Monte-Carlo Tree Search (MCTS) con selección UCT para juegos de dos jugadores alternados.

El agente utiliza simulaciones para estimar qué acciones tienen mayor probabilidad de conducir a estados favorables, balanceando exploración y explotación mediante UCB1.

El proyecto incluye dos versiones:

- `AnaPolicy`: reconstruye el árbol en cada turno.
- `AnaPolicyPersistent`: reutiliza estadísticas del árbol durante una partida.

---

## Conceptos implementados

- MCTS/UCT
- Juegos alternados de dos jugadores
- Reward shaping
- Persistencia online del árbol
- Heurísticas de victoria/bloqueo inmediato

---

## Estructura

```text
groups/Ana/
├── policy.py       # Implementación de los agentes
├── mcts.py         # Algoritmo MCTS/UCT
├── README.md
└── entrega.ipynb   # Validación experimental y análisis
```

---

## Ejecución

Desde la raíz del repositorio:

```bash
python main.py
```

## Torneos

```bash
python run_tournament.py
```

La versión `AnaPolicyPersistent` fue diseñada como candidata principal para el torneo debido a la reutilización de estadísticas entre turnos.

Los resultados se guardan en:

```text
versus/tournament_ana_results.json
```

---

## Uso del Agente

```python
from groups.Ana.policy import AnaPolicy

policy = AnaPolicy(num_simulations=200)
policy.mount(timeout=5)

action = policy.act(board)
```
También está disponible:

```python
AnaPolicyPersistent
```

---

## Parámetros principales

- `num_simulations`: número de simulaciones MCTS por turno.
- `reward_shaping`: activa bonificación por amenazas intermedias.

---

## Notebook de Análisis

El archivo `entrega.ipynb` contiene:

- validación empírica,
- comparación entre versiones,
- análisis de parámetros,
- gráficas,
- conclusiones,
- propuestas de mejora.

---

## Resultados Generales

Los experimentos muestran que ambas versiones superan consistentemente al agente aleatorio y que el desempeño mejora al aumentar el número de simulaciones.

La persistencia del árbol modifica el comportamiento del agente entre turnos y puede aportar ventajas especialmente cuando el presupuesto de simulaciones es limitado.

---

## Requisitos

Dependencias principales:

- numpy
- matplotlib
- pandas

El agente fue probado utilizando Python 3.10.