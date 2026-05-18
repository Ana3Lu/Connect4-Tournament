import numpy as np
from connect4.connect_state import ConnectState
from groups.Ana.policy import AnaPolicy, AnaPolicyPersistent


class RandomPolicy:
    def mount(self): self._rng = np.random.default_rng()
    def act(self, s): return int(self._rng.choice([c for c in range(7) if s[0, c] == 0]))


def play_game(policy_a, policy_b, seed=None):
    rng = np.random.default_rng(seed)
    state = ConnectState()
    policies = {-1: policy_a, 1: policy_b}
    while not state.is_final():
        action = policies[state.player].act(state.board)
        state = state.transition(int(action))
    return state.get_winner()


def run(agent_cls, kwargs, n=20):
    wins, losses, draws = 0, 0, 0
    for i in range(n):
        a = agent_cls(**kwargs); a.mount()
        r = RandomPolicy();      r.mount()
        # alternar quién va primero
        if i % 2 == 0:
            w = play_game(a, r, seed=i)
            if w == -1: wins += 1
            elif w == 1: losses += 1
            else: draws += 1
        else:
            w = play_game(r, a, seed=i)
            if w == 1: wins += 1
            elif w == -1: losses += 1
            else: draws += 1
    print(f"Wins: {wins} | Losses: {losses} | Draws: {draws} | Win rate: {wins/n:.0%}")


print("V1 (num_simulations=200):")
run(AnaPolicy, {"num_simulations": 200})

print("V2 persistente (num_simulations=200):")
run(AnaPolicyPersistent, {"num_simulations": 200})
