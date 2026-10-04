"""Paired seeded synthetic experiments. No LLM API calls in benchmark mode."""
from dataclasses import asdict
import math
from .agent import Config, DoctorAgent
from .simulator import SyntheticPatient, demo_model


def wilson(successes: int, total: int) -> list[float] | None:
    if not total:
        return None
    p, z = successes / total, 1.96
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [round(max(0, center - half), 4), round(min(1, center + half), 4)]


def benchmark(n: int = 1000, seed: int = 10000, max_actions: int = 60) -> dict:
    if n < 1:
        raise ValueError("n must be positive")
    model = demo_model()
    report = {"scope": "synthetic_only_not_clinical_validation", "cases": n,
              "seed_start": seed, "seed_end": seed + n - 1, "max_actions": max_actions,
              "config": asdict(Config(max_actions=max_actions)),
              "model": asdict(model) | {"urgent": sorted(model.urgent)},
              "model_assumption": "invented independent binary likelihoods shared by simulator and agent",
              "policies": {}}
    for policy in ("random", "information"):
        counts = dict(diagnosed=0, correct_diagnosed=0, urgent_evaluation=0, inconclusive=0,
                      urgent_cases=0, urgent_escalated=0, urgent_benign_diagnosed=0,
                      nonurgent_escalated=0, budget_violations=0, duplicate_violations=0)
        steps, costs, errors = [], [], []
        for case_seed in range(seed, seed + n):
            patient = SyntheticPatient(model, case_seed)
            result = DoctorAgent(model, Config(max_actions=max_actions, policy=policy, seed=case_seed)).run(patient)
            counts[result.status] += 1
            urgent = patient.truth in model.urgent
            counts["urgent_cases"] += urgent
            counts["urgent_escalated"] += urgent and result.status == "urgent_evaluation"
            counts["nonurgent_escalated"] += not urgent and result.status == "urgent_evaluation"
            counts["correct_diagnosed"] += result.status == "diagnosed" and result.diagnosis == patient.truth
            counts["urgent_benign_diagnosed"] += urgent and result.status == "diagnosed"
            counts["budget_violations"] += result.attempts > max_actions
            counts["duplicate_violations"] += len(patient.performed) != len(set(patient.performed))
            steps.append(result.attempts)
            costs.append(result.cost)
            if (result.status == "diagnosed" and result.diagnosis != patient.truth) or (urgent and result.status != "urgent_evaluation"):
                if len(errors) < 10:
                    errors.append({"seed": case_seed, "truth": patient.truth, "result": asdict(result)})
        counts["urgent_not_escalated"] = counts["urgent_cases"] - counts["urgent_escalated"]
        ordered = sorted(steps)
        report["policies"][policy] = {
            "counts": counts,
            "mean_actions": round(sum(steps) / n, 3),
            "p95_actions": ordered[math.ceil(.95 * n) - 1],
            "mean_cost": round(sum(costs) / n, 3),
            "diagnostic_coverage": counts["diagnosed"] / n,
            "accuracy_when_diagnosed": counts["correct_diagnosed"] / counts["diagnosed"] if counts["diagnosed"] else None,
            "accuracy_when_diagnosed_ci95": wilson(counts["correct_diagnosed"], counts["diagnosed"]),
            "urgent_escalation_recall": counts["urgent_escalated"] / counts["urgent_cases"] if counts["urgent_cases"] else None,
            "urgent_escalation_recall_ci95": wilson(counts["urgent_escalated"], counts["urgent_cases"]),
            "nonurgent_escalation_rate": counts["nonurgent_escalated"] / (n - counts["urgent_cases"]) if n != counts["urgent_cases"] else None,
            "failure_examples": errors,
        }
    return report
