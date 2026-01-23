from collections import defaultdict, Counter
import random


class MarkovChain:
    def __init__(self, order=2):
        self.order = order
        self.model = defaultdict(Counter)
        self._keys_cache = None

    def train(self, sequences):
        for seq in sequences:
            if isinstance(seq, str):
                tokens = seq.strip().split()
            else:
                tokens = seq
            if len(tokens) < self.order + 1:
                continue
            for i in range(len(tokens) - self.order):
                state = tuple(tokens[i:i+self.order])
                next_token = tokens[i+self.order]
                self.model[state][next_token] += 1
        self._keys_cache = None # Invalidate cache after training

    def generate(self, length=32, seed=None, start_token=None):
        if not self.model:
            raise ValueError("Model is empty. Train first.")

        # Lazily create and cache the list of keys for performance
        if not hasattr(self, '_keys_cache') or self._keys_cache is None:
            self._keys_cache = list(self.model.keys())
        
        if not self._keys_cache:
            return [] # Model was trained but resulted in no transitions

        # Determine the starting state for generation
        if seed:
            current_state = tuple(seed)
        elif start_token:
            # For order 1, the state is simply the previous token
            if self.order == 1:
                current_state = (start_token,)
            else:
                # For higher orders, find a known state that ends with the previous token
                # to ensure a valid transition.
                possible_starts = [s for s in self._keys_cache if s[-1] == start_token]
                if possible_starts:
                    current_state = random.choice(possible_starts)
                else:
                    # Fallback to a random state if no valid transition from start_token exists
                    current_state = random.choice(self._keys_cache)
        else:
            # No seed or start token, pick a random state
            current_state = random.choice(self._keys_cache)

        result = []
        
        for _ in range(length):
            next_tokens = self.model.get(current_state)
            
            # If we hit a dead end (the current state has no recorded next tokens)
            if not next_tokens:
                # Jump to a new random state to continue generation
                current_state = random.choice(self._keys_cache)
                next_tokens = self.model.get(current_state)
                if not next_tokens:
                    break # If even the random state is a dead end, stop.

            next_token = random.choices(
                list(next_tokens.keys()),
                weights=list(next_tokens.values())
            )[0]
            
            result.append(next_token)
            
            # Update the state for the next iteration
            current_state = (*current_state[1:], next_token)
            
        return result