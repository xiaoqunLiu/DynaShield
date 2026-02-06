"""
Character-level perturbation functions for SmoothLLM defense.
Based on: https://github.com/arobey1/smooth-llm
"""

import random
import string


class Perturbation:
    """Base class for random perturbations."""

    def __init__(self, q):
        """
        Args:
            q: Perturbation percentage (0-100)
        """
        self.q = q
        self.alphabet = string.printable


class RandomSwapPerturbation(Perturbation):
    """Implementation of random swap perturbations.
    
    Randomly replaces q% of characters in the string with random characters
    from the printable alphabet.
    """

    def __init__(self, q):
        super(RandomSwapPerturbation, self).__init__(q)

    def __call__(self, s):
        """Apply random swap perturbation to string s."""
        list_s = list(s)
        num_chars_to_perturb = int(len(s) * self.q / 100)
        if num_chars_to_perturb == 0:
            return s
        sampled_indices = random.sample(range(len(s)), num_chars_to_perturb)
        for i in sampled_indices:
            list_s[i] = random.choice(self.alphabet)
        return ''.join(list_s)


class RandomPatchPerturbation(Perturbation):
    """Implementation of random patch perturbations.
    
    Replaces a contiguous substring of length q% with random characters.
    """

    def __init__(self, q):
        super(RandomPatchPerturbation, self).__init__(q)

    def __call__(self, s):
        """Apply random patch perturbation to string s."""
        list_s = list(s)
        substring_width = int(len(s) * self.q / 100)
        if substring_width == 0:
            return s
        max_start = len(s) - substring_width
        start_index = random.randint(0, max_start)
        sampled_chars = ''.join([
            random.choice(self.alphabet) for _ in range(substring_width)
        ])
        list_s[start_index:start_index+substring_width] = sampled_chars
        return ''.join(list_s)


class RandomInsertPerturbation(Perturbation):
    """Implementation of random insert perturbations.
    
    Inserts random characters at q% of positions in the string.
    """

    def __init__(self, q):
        super(RandomInsertPerturbation, self).__init__(q)

    def __call__(self, s):
        """Apply random insert perturbation to string s."""
        list_s = list(s)
        num_chars_to_insert = int(len(s) * self.q / 100)
        if num_chars_to_insert == 0:
            return s
        sampled_indices = random.sample(range(len(s)), num_chars_to_insert)
        # Sort indices in reverse to maintain correct positions during insertion
        sampled_indices.sort(reverse=True)
        for i in sampled_indices:
            list_s.insert(i, random.choice(self.alphabet))
        return ''.join(list_s)
