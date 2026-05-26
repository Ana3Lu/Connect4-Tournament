# Agente Connect-4: Q-Learning Tabular

## Idea principal

Este agente aplica **Q-learning tabular** para aprender a jugar Connect-4.
Durante `mount()` entrena offline haciendo *self-play* (partidas contra sí mismo),
construyendo una tabla Q que mapea estados del tablero a valores de acción.
Durante `act()` combina cuatro niveles de decisión:

1. **Ganar en 1 movimiento** — si existe una jugada ganadora inmediata, la toma.
2. **Bloquear al rival** — si el rival puede ganar en el siguiente turno, lo bloquea.
3. **Tabla Q** — si el estado fue visitado durante el entrenamiento, elige la columna con mayor Q-valor.
4. **Heurística de fallback** — si el estado es desconocido, evalúa ventanas de 4 celdas para estimar la mejor jugada.

## Fundamento matemático

El agente actualiza sus valores con la ecuación de Bellman aplicada a Q-learning:

```
Q(s,a) ← Q(s,a) + α · [ r + γ · max_a' Q(s',a') − Q(s,a) ]
```

Donde el término entre corchetes es el **error TD** — la diferencia entre
lo que el agente esperaba ganar y lo que realmente ganó.

## Parámetros configurables

| Parámetro | Descripción | Valor |
|---|---|---|
| `episodes` | Partidas de entrenamiento | 30,000 |
| `alpha` | Tasa de aprendizaje | 0.1 |
| `gamma` | Factor de descuento | 0.95 |
| `epsilon_start` | Exploración inicial | 1.0 |
| `epsilon_end` | Exploración mínima | 0.05 |
| `save_path` | Ruta para guardar/cargar la tabla Q | `q_table.pkl` |

## Uso rápido

```python
from policy import QLearningAgent

agent = QLearningAgent(episodes=30_000)
agent.mount()          # entrena y guarda tabla Q
col = agent.act(board) # board: np.ndarray (6, 7)
```

## Estructura de archivos

```
Juliana/
├── policy.py           # Agente principal (QLearningAgent)
├── entrega.ipynb       # Notebook con análisis completo
├── readme.md           # Este archivo
└── q_table_*.pkl       # Tablas Q pre-entrenadas
```

## Versiones del agente

| Versión | Descripción |
|---|---|
| **v1** `QLearningNoHeuristic` | Sin heurística de fallback — aleatorio si estado desconocido |
| **v2** `QLearningAgent` (final) | Con heurística de fallback — versión entregada |

## Resultados

- Nunca pierde contra el jugador aleatorio para ambos colores
- Win rate vs aleatorio: **99.7%** (v2) vs **98.0%** (v1)
- Self-play converge al **50%** — comportamiento simétrico esperado
- Estados aprendidos con 30,000 episodios: **460,794**

## Propuestas de mejora

| Limitación | Propuesta |
|---|---|
| Solo cubre 0.0001% del espacio de estados | Deep Q-Network (DQN) |
| Recompensa solo al final de la partida | Reward shaping con heurística intermedia |
| Entrenado solo contra sí mismo | Curriculum learning contra oponentes variados |
