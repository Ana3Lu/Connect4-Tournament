# AnaPolicy — MCTS/UCT incremental para Connect-4

## Descripción

**AnaPolicy** es un agente para Connect-4 basado en Monte-Carlo Tree Search (MCTS) con selección UCT para juegos alternados de dos jugadores.

El proyecto fue desarrollado de manera incremental con el objetivo de estudiar cómo distintos mecanismos modifican el comportamiento y desempeño de un agente MCTS/UCT.

A lo largo del desarrollo se incorporaron múltiples mejoras sobre una arquitectura base, incluyendo:

- reward shaping,
- persistencia intra-partida del árbol,
- heurísticas tácticas,
- rollouts inteligentes.

El agente final fue diseñado para superar consistentemente al agente aleatorio oficial del torneo y mantener estabilidad bajo distintos presupuestos de simulación.

---

## Evolución incremental del agente

| Versión | Características principales |
|---|---|
| `AnaPolicyV1` | MCTS/UCT base |
| `AnaPolicyV2` | + reward shaping |
| `AnaPolicyFinal` | + persistencia + heurísticas tácticas + rollout inteligente |

La evolución del agente se realizó de manera incremental, incorporando progresivamente mecanismos adicionales sobre una arquitectura MCTS reutilizable.

---

## Conceptos implementados

- Monte-Carlo Tree Search (MCTS)
- Upper Confidence Bound applied to Trees (UCT)
- Juegos alternados de dos jugadores
- Reward shaping
- Persistencia intra-partida del árbol
- Rollout inteligente
- Heurísticas tácticas
- Detección de victoria inmediata
- Bloqueo urgente de amenazas
- Reutilización online de estadísticas del árbol

---

## Arquitectura general

El agente utiliza el flujo clásico de MCTS:

1. **Selection**
   - selección mediante UCT,
   - balance exploración/explotación.

2. **Expansion**
   - expansión incremental del árbol.

3. **Simulation**
   - rollouts aleatorios o inteligentes.

4. **Backpropagation**
   - propagación alternante de recompensas.

La implementación fue diseñada para permitir activar o desactivar distintos mecanismos experimentales sin modificar la arquitectura base.

---

## Estructura del proyecto

```text
groups/Ana/
├── policy.py
├── mcts.py
├── README.md
└── entrega.ipynb
```

### Archivos principales

#### `policy.py`

Implementa las distintas versiones del agente:

- `AnaPolicyV1`
- `AnaPolicyV2`
- `AnaPolicyFinal`

También contiene:

- heurísticas tácticas,
- lógica de configuración,
- integración con el framework del torneo.

#### `mcts.py`

Implementa el núcleo configurable de MCTS/UCT:

- selección UCT,
- rollouts,
- reward shaping,
- persistencia opcional,
- estadísticas del árbol.

#### `entrega.ipynb`

Contiene:

- validación experimental,
- análisis empírico,
- gráficas,
- comparación entre mecanismos,
- análisis de sensibilidad,
- discusión metodológica.

---

## Ejecución

Desde la raíz del repositorio:

```bash
python main.py
```

---

## Uso de los agentes

```python
from groups.Ana.policy import (
    AnaPolicyV1,
    AnaPolicyV2,
    AnaPolicyFinal,
)

policy = AnaPolicyFinal(
    num_simulations=200
)

policy.mount(timeout=5)

action = policy.act(board)
```

---

## Versiones implementadas

### AnaPolicyV1

Versión base:

- MCTS + UCT estándar,
- rollout aleatorio,
- sin shaping,
- sin persistencia.

---

### AnaPolicyV2

Agrega reward shaping:

- bonificaciones intermedias,
- evaluación de amenazas de 3 en línea,
- mejora parcial de exploración.

---

### AnaPolicyFinal

Integra múltiples mejoras adicionales:

- persistencia intra-partida,
- reutilización de estadísticas,
- heurísticas tácticas,
- victoria inmediata,
- bloqueo urgente,
- rollout inteligente.

Esta versión corresponde al agente competitivo final utilizado en la evaluación principal.

---

## Parámetros principales

| Parámetro | Descripción |
|---|---|
| `num_simulations` | Número de simulaciones MCTS por turno |
| `reward_shaping` | Activa señales intermedias durante rollouts |
| `persistent_tree` | Reutiliza estadísticas entre turnos |
| `smart_rollout` | Prioriza jugadas tácticas durante simulaciones |

---

## Notebook de análisis

El notebook `entrega.ipynb` incluye:

- validación contra agente aleatorio,
- análisis incremental de mecanismos,
- impacto del número de simulaciones,
- comparación entre versiones,
- análisis de persistencia,
- evaluación de reward shaping,
- validación de heurísticas tácticas,
- métricas de desempeño,
- discusión de limitaciones.

---

## Resultados generales

Los experimentos muestran que:

- todas las versiones superan consistentemente al agente aleatorio propuesto,
- el reward shaping mejora la calidad de exploración,
- la persistencia reutiliza experiencia intra-partida,
- las heurísticas tácticas mejoran la respuesta ante amenazas inmediatas,
- el desempeño aumenta al incrementar el presupuesto de simulaciones.

Los resultados también evidencian que las mejoras tácticas fueron las más determinantes para alcanzar comportamiento competitivo estable.

---

## Limitaciones identificadas

El análisis experimental permitió identificar varias limitaciones:

- manejo incompleto de forks múltiples,
- rollouts todavía parcialmente aleatorios,
- constante de exploración fija,
- ausencia de aprendizaje entre partidas.

Estas limitaciones dejan espacio para futuras mejoras sobre la arquitectura actual.

---

## Requisitos

Dependencias principales:

- numpy
- matplotlib
- pandas

El proyecto fue probado utilizando:

- Python 3.10