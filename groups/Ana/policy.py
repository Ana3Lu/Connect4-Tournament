import numpy as np
from connect4.policy import Policy


class AnaPolicy(Policy):

    def mount(self) -> None:
        pass

    def act(self, s: np.ndarray) -> int:
        rng = np.random.default_rng()
        return int(rng.choice([c for c in range(7) if s[0, c] == 0]))
