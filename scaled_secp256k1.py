from __future__ import annotations

import math
import random
from dataclasses import dataclass


Point = tuple[int, int] | None


@dataclass
class CollisionResult:
    """Measured and theoretical collision probabilities for one curve."""

    b: int
    p: int
    n: int
    population_size: int
    trials: int
    collisions: int

    @property
    def theoretical_probability(self) -> float:
        return self.population_size / (self.n - 1)

    @property
    def experimental_probability(self) -> float:
        return self.collisions / self.trials

    @property
    def log2_theoretical_probability(self) -> float:
        return math.log2(self.theoretical_probability)

    @property
    def log2_experimental_probability(self) -> float:
        if self.collisions == 0:
            return float("-inf")
        return math.log2(self.experimental_probability)


class ScaledSecp256k1:
    """A small prime-order analogue of secp256k1 for 4 <= b <= 40.

    The point count uses the CM structure of y^2 = x^3 + 7 (j = 0),
    so it does not enumerate all p^2 coordinate pairs.
    """

    _MR_BASES_64 = (2, 325, 9375, 28178, 450775, 9780504, 1795265022)

    def __init__(self, b: int, seed: int = 42):
        if not 4 <= b <= 40:
            raise ValueError("This educational implementation expects 4 <= b <= 40")
        self.b = b
        self.p: int | None = None
        self.n: int | None = None
        self.generator: Point = None
        self.rng = random.Random(seed)

    @classmethod
    def is_prime(cls, value: int) -> bool:
        """Deterministic Miller-Rabin test for value < 2^64."""
        if value < 2:
            return False
        small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
        for prime in small_primes:
            if value % prime == 0:
                return value == prime

        d = value - 1
        s = 0
        while d % 2 == 0:
            s += 1
            d //= 2

        for base in cls._MR_BASES_64:
            if base % value == 0:
                continue
            x = pow(base, d, value)
            if x in (1, value - 1):
                continue
            for _ in range(s - 1):
                x = x * x % value
                if x == value - 1:
                    break
            else:
                return False
        return True

    @staticmethod
    def _trace_candidates(p: int) -> set[int]:
        """Return the six possible Frobenius traces for a j=0 curve."""
        limit = math.isqrt((4 * p) // 27)
        for m in range(1, limit + 1):
            remainder = 4 * p - 27 * m * m
            ell = math.isqrt(remainder)
            if ell * ell == remainder:
                return {
                    ell,
                    -ell,
                    (ell + 3 * m) // 2,
                    -(ell + 3 * m) // 2,
                    (ell - 3 * m) // 2,
                    -(ell - 3 * m) // 2,
                }
        raise ArithmeticError(f"Could not represent 4p = L^2 + 27M^2 for p={p}")

    @staticmethod
    def _sqrt_mod(value: int, p: int) -> int | None:
        """Tonelli-Shanks square root modulo an odd prime."""
        value %= p
        if value == 0:
            return 0
        if pow(value, (p - 1) // 2, p) != 1:
            return None
        if p % 4 == 3:
            return pow(value, (p + 1) // 4, p)

        q = p - 1
        s = 0
        while q % 2 == 0:
            s += 1
            q //= 2

        z = 2
        while pow(z, (p - 1) // 2, p) != p - 1:
            z += 1

        m = s
        c = pow(z, q, p)
        t = pow(value, q, p)
        r = pow(value, (q + 1) // 2, p)
        while t != 1:
            i = 1
            t2i = t * t % p
            while t2i != 1:
                t2i = t2i * t2i % p
                i += 1
            factor = pow(c, 1 << (m - i - 1), p)
            r = r * factor % p
            t = t * factor * factor % p
            c = factor * factor % p
            m = i
        return r

    @staticmethod
    def add(left: Point, right: Point, p: int) -> Point:
        """Add two points; None denotes the point at infinity."""
        if left is None:
            return right
        if right is None:
            return left

        x1, y1 = left
        x2, y2 = right
        if x1 == x2 and (y1 + y2) % p == 0:
            return None

        if left == right:
            slope = 3 * x1 * x1 * pow(2 * y1, -1, p) % p
        else:
            slope = (y2 - y1) * pow(x2 - x1, -1, p) % p

        x3 = (slope * slope - x1 - x2) % p
        y3 = (slope * (x1 - x3) - y1) % p
        return x3, y3

    @classmethod
    def multiply(cls, scalar: int, point: Point, p: int) -> Point:
        """Compute scalar * point with double-and-add."""
        result: Point = None
        addend = point
        while scalar:
            if scalar & 1:
                result = cls.add(result, addend, p)
            addend = cls.add(addend, addend, p)
            scalar >>= 1
        return result

    def _random_point(self, p: int) -> Point:
        while True:
            x = self.rng.randrange(p)
            y = self._sqrt_mod((x * x % p) * x + 7, p)
            if y is not None:
                return x, y

    def find_parameters(self) -> tuple[int, int]:
        """Find p and prime #E(F_p)=n, both with exactly b bits."""
        lower = 1 << (self.b - 1)
        upper = 1 << self.b
        p = lower + ((1 - lower) % 6)

        while p < upper:
            if self.is_prime(p):
                point = self._random_point(p)
                for trace in self._trace_candidates(p):
                    n = p + 1 - trace
                    if lower <= n < upper and self.is_prime(n):
                        if self.multiply(n, point, p) is None:
                            self.p = p
                            self.n = n
                            self.generator = point
                            return p, n
            p += 6
        raise RuntimeError(f"No suitable curve parameters found for b={self.b}")

    def collision_experiment(self, population_size: int, trials: int) -> CollisionResult:
        """Generate N used keys and measure collisions over the requested trials."""
        if self.n is None or self.p is None:
            self.find_parameters()
        assert self.n is not None and self.p is not None
        if not 0 < population_size < self.n:
            raise ValueError("population_size must satisfy 0 < N < n")
        if trials <= 0:
            raise ValueError("trials must be positive")

        existing_keys = set(self.rng.sample(range(1, self.n), population_size))
        collisions = sum(
            self.rng.randrange(1, self.n) in existing_keys
            for _ in range(trials)
        )
        return CollisionResult(
            b=self.b,
            p=self.p,
            n=self.n,
            population_size=population_size,
            trials=trials,
            collisions=collisions,
        )
