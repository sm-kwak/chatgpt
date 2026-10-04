from dataclasses import dataclass
from enum import Enum
from typing import Protocol
import math


class Kind(str, Enum):
    QUESTION = "question"
    TEST = "test"
    EXAM = "physical_exam"


@dataclass(frozen=True)
class Action:
    id: str
    kind: Kind
    prompt: str
    cost: float
    positive_likelihood: tuple[float, ...]
    safety_screen: bool = False


@dataclass(frozen=True)
class Observation:
    action_id: str
    positive: bool


@dataclass(frozen=True)
class Model:
    diagnoses: tuple[str, ...]
    priors: tuple[float, ...]
    urgent: frozenset[str]
    actions: tuple[Action, ...]
    required_before_benign: tuple[str, ...] = ()

    def __post_init__(self):
        n = len(self.diagnoses)
        if n < 2 or len(set(self.diagnoses)) != n:
            raise ValueError("Need at least two unique diagnoses")
        if len(self.priors) != n or any(not math.isfinite(p) or p <= 0 for p in self.priors):
            raise ValueError("Priors must be finite and positive")
        if not math.isclose(sum(self.priors), 1.0):
            raise ValueError("Priors must sum to one")
        if not self.urgent <= set(self.diagnoses):
            raise ValueError("Unknown urgent diagnosis")
        if not self.actions or len({a.id for a in self.actions}) != len(self.actions):
            raise ValueError("Actions must have unique IDs")
        if not set(self.required_before_benign) <= {a.id for a in self.actions}:
            raise ValueError("Unknown required exclusion action")
        for a in self.actions:
            if not a.id or not isinstance(a.kind, Kind) or not math.isfinite(a.cost) or a.cost <= 0:
                raise ValueError("Invalid atomic action")
            if len(a.positive_likelihood) != n or any(not 0 < p < 1 for p in a.positive_likelihood):
                raise ValueError("Likelihoods must be strictly between zero and one")


class PatientEnvironment(Protocol):
    """Implement with a simulator/benchmark; never expose hidden labels to the agent.

    One call performs ONE catalogued question, test, or examination.
    Failed calls consume an attempt; never silently retry a patient action.
    """

    def perform(self, action: Action) -> Observation: ...


def update(prior: tuple[float, ...], action: Action, positive: bool) -> tuple[float, ...]:
    weights = tuple(p * (q if positive else 1 - q) for p, q in zip(prior, action.positive_likelihood))
    total = sum(weights)
    return tuple(w / total for w in weights)


def entropy(probs: tuple[float, ...]) -> float:
    return -sum(p * math.log2(p) for p in probs if p > 0)


def information_gain(prior: tuple[float, ...], action: Action) -> float:
    positive = sum(p * q for p, q in zip(prior, action.positive_likelihood))
    expected = positive * entropy(update(prior, action, True))
    expected += (1 - positive) * entropy(update(prior, action, False))
    return max(0.0, entropy(prior) - expected)
