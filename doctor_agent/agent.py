from dataclasses import dataclass, field
import random
from typing import Protocol
from .domain import Action, Model, Observation, PatientEnvironment, information_gain, update


class Adviser(Protocol):
    def choose(self, posterior: dict[str, float], history: list[Observation], candidates: list[Action]) -> str | None: ...


@dataclass(frozen=True)
class Config:
    max_actions: int = 60
    confidence: float = .95
    max_benign_urgent_risk: float = .005
    policy: str = "information"
    seed: int = 0

    def __post_init__(self):
        if type(self.max_actions) is not int or not 1 <= self.max_actions <= 60:
            raise ValueError("max_actions must be an integer between 1 and 60")
        if not .5 < self.confidence < 1 or not 0 < self.max_benign_urgent_risk < .5:
            raise ValueError("Invalid confidence/risk limits")
        if self.policy not in ("information", "random"):
            raise ValueError("Unknown policy")


@dataclass
class Result:
    status: str
    reason: str
    diagnosis: str | None
    posterior: dict[str, float]
    attempts: int
    cost: float
    history: list[dict] = field(default_factory=list)


class DoctorAgent:
    def __init__(self, model: Model, config: Config | None = None, adviser: Adviser | None = None):
        self.model = model
        self.config = config or Config()
        self.adviser = adviser

    def run(self, environment: PatientEnvironment) -> Result:
        posterior = self.model.priors
        observations: list[Observation] = []
        history: list[dict] = []
        attempted: set[str] = set()
        cost = 0.0
        rng = random.Random(self.config.seed)

        def finish(status: str, reason: str, diagnosis: str | None = None) -> Result:
            return Result(status, reason, diagnosis, dict(zip(self.model.diagnoses, posterior)), len(attempted), cost, history)

        while True:
            remaining = [a for a in self.model.actions if a.id not in attempted]
            screens = [a for a in remaining if a.safety_screen]
            top = max(range(len(posterior)), key=posterior.__getitem__)
            diagnosis = self.model.diagnoses[top]
            urgent_risk = sum(p for d, p in zip(self.model.diagnoses, posterior) if d in self.model.urgent)
            exclusions = [a for a in remaining if a.id in self.model.required_before_benign]
            if not screens and posterior[top] >= self.config.confidence:
                if diagnosis in self.model.urgent:
                    return finish("urgent_evaluation", "urgent_diagnosis_suspected", diagnosis)
                if not exclusions and urgent_risk <= self.config.max_benign_urgent_risk:
                    return finish("diagnosed", "confidence_and_risk_thresholds_met", diagnosis)
            if len(attempted) >= self.config.max_actions:
                if urgent_risk > self.config.max_benign_urgent_risk:
                    return finish("urgent_evaluation", "action_budget_exhausted_with_unresolved_urgent_risk")
                return finish("inconclusive", "action_budget_exhausted")
            if not remaining:
                if urgent_risk > self.config.max_benign_urgent_risk:
                    return finish("urgent_evaluation", "no_remaining_actions_with_unresolved_urgent_risk")
                return finish("inconclusive", "no_remaining_actions")

            source = "safety_screen"
            if screens:
                selected = screens[0]
            elif exclusions and diagnosis not in self.model.urgent and posterior[top] >= self.config.confidence:
                selected = max(exclusions, key=lambda a: information_gain(posterior, a) / a.cost)
                source = "required_exclusion"
            elif self.config.policy == "random":
                selected = rng.choice(remaining)
                source = "random"
            else:
                ranked = sorted(remaining, key=lambda a: (-information_gain(posterior, a) / a.cost, a.id))
                selected = ranked[0]
                source = "information_per_cost"
                if self.adviser:
                    # LLM proposes only one available action; terminal decisions remain local.
                    candidates = ranked[:3]
                    try:
                        proposal = self.adviser.choose(dict(zip(self.model.diagnoses, posterior)), list(observations), candidates)
                    except Exception:
                        proposal = None
                    match = next((a for a in candidates if a.id == proposal), None)
                    if match is not None:
                        selected, source = match, "llm"
                    else:
                        source += "_llm_fallback"

            # Reserve budget BEFORE invocation, including failed or malformed responses.
            attempted.add(selected.id)
            cost += selected.cost
            entry = {"step": len(attempted), "action_id": selected.id, "kind": selected.kind.value,
                     "source": source, "cost": selected.cost}
            history.append(entry)
            try:
                observation = environment.perform(selected)
                if not isinstance(observation, Observation) or observation.action_id != selected.id or type(observation.positive) is not bool:
                    raise ValueError("Invalid observation")
            except Exception:
                entry["error"] = "environment_failure_or_invalid_observation"
                # Do not leak provider responses/secrets or silently retry clinical actions.
                return finish("inconclusive", "environment_error")
            entry["positive"] = observation.positive
            observations.append(observation)
            posterior = update(posterior, selected, observation.positive)
            entry["posterior_after"] = dict(zip(self.model.diagnoses, posterior))
            if selected.safety_screen and observation.positive:
                return finish("urgent_evaluation", "positive_safety_screen")
