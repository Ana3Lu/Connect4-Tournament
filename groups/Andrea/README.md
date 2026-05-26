# AndreUCB1 — UCB1 + Minimax para Connect-4

## Descripción

**AndreUCB1** es un agente para Connect-4 que combina reglas determinísticas, Minimax con poda alpha-beta y UCB1 como criterio de desempate.

El agente garantiza que **nunca pierde contra un jugador aleatorio**, gracias a sus tres capas de decisión:

1. **Victoria inmediata** — si existe una jugada ganadora, la toma.
2. **Bloqueo** — si el oponente tiene victoria inmediata, la bloquea.
3. **Minimax + UCB1** — evaluación táctica del tablero a profundidad 4.

---

## Estructura

```text
groups/Andrea/
├── policy.py   # Implementación del agente AndreUCB1
└── README.md
```

---

## Ejecución

Desde la raíz del repositorio:

```bash
python main.py
```

Torneo completo:

```bash
python run_tournament.py
```

---

## Uso del Agente

```python
from groups.Andrea.policy import AndreUCB1

policy = AndreUCB1()
policy.mount()

action = policy.act(board)  # board: np.ndarray (6x7)
```

---

## Parámetros Principales

| Parámetro   | Valor por defecto | Descripción                          |
|-------------|-------------------|--------------------------------------|
| `DEPTH`     | `4`               | Profundidad del árbol Minimax        |
| `COL_ORDER` | `[3,2,4,1,5,0,6]` | Orden de exploración (centro primero)|
| `ROWS`      | `6`               | Filas del tablero                    |
| `COLS`      | `7`               | Columnas del tablero                 |

---

## Requisitos

- Python 3.10+
- numpy

```bash
pip install numpy
```
