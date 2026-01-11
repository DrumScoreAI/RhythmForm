from collections import defaultdict, Counter
import random


class MarkovChain:
    def __init__(self, order=2):
        self.order = order
        self.model = defaultdict(Counter)

    def train(self, sequences):
        for seq in sequences:
            tokens = seq.strip().split()
            if len(tokens) < self.order + 1:
                continue
            for i in range(len(tokens) - self.order):
                state = tuple(tokens[i:i+self.order])
                next_token = tokens[i+self.order]
                self.model[state][next_token] += 1

    def generate(self, length=32, seed=None, start_token=None):
        if not self.model:
            raise ValueError("Model is empty. Train first.")
        state = random.choice(list(self.model.keys())) if seed is None else seed
        result = list(state)
        if start_token:
            result[0] = start_token
            state = (start_token,) + state[1:]
        for _ in range(length - self.order):
            next_tokens = self.model.get(tuple(state), None)
            if not next_tokens:
                break
            next_token = random.choices(
                list(next_tokens.keys()),
                weights=list(next_tokens.values())
            )[0]
            result.append(next_token)
            state = (*state[1:], next_token)
        return result