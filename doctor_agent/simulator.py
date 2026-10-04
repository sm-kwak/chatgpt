"""Invented likelihoods; conditional independence is a toy assumption, NOT clinical evidence."""
import random
from .domain import Action, Kind, Model, Observation


def demo_model() -> Model:
    # Order: ACS, PE, pneumonia, reflux, panic. Costs are arbitrary simulator units.
    return Model(
        diagnoses=("acute_coronary_syndrome", "pulmonary_embolism", "pneumonia", "reflux", "panic"),
        priors=(0.15, 0.10, 0.20, 0.30, 0.25),
        urgent=frozenset({"acute_coronary_syndrome", "pulmonary_embolism"}),
        actions=(
            Action("systolic_bp_low", Kind.EXAM, "Measure systolic blood pressure: below 90 mmHg?", 1, (.30, .25, .10, .01, .01), True),
            Action("severe_dyspnea", Kind.QUESTION, "Are you severely short of breath at rest?", 1, (.25, .55, .20, .01, .05), True),
            Action("exertional_pain", Kind.QUESTION, "Does exertion trigger the chest pain?", 1, (.80, .30, .15, .10, .15)),
            Action("pleuritic_pain", Kind.QUESTION, "Does breathing in worsen the pain?", 1, (.10, .80, .65, .05, .10)),
            Action("meal_related", Kind.QUESTION, "Does pain follow meals?", 1, (.10, .05, .05, .85, .10)),
            Action("fear_episode", Kind.QUESTION, "Did sudden intense fear accompany this episode?", 1, (.20, .20, .05, .05, .85)),
            Action("cough", Kind.QUESTION, "Do you have a cough?", 1, (.05, .15, .85, .10, .05)),
            Action("temperature_high", Kind.EXAM, "Measure temperature: above 38 C?", 1.5, (.05, .10, .90, .02, .02)),
            Action("unilateral_calf_tenderness", Kind.EXAM, "Is the left calf tender on palpation?", 2, (.05, .65, .05, .02, .02)),
            Action("troponin_elevated", Kind.TEST, "Obtain a troponin result: elevated?", 5, (.95, .15, .08, .01, .01)),
            Action("d_dimer_elevated", Kind.TEST, "Obtain a D-dimer result: elevated?", 5, (.20, .95, .40, .08, .08)),
            Action("xray_infiltrate", Kind.TEST, "Obtain chest radiograph: focal infiltrate present?", 6, (.05, .08, .95, .01, .01)),
            Action("ecg_ischemic", Kind.TEST, "Obtain ECG: ischemic change present?", 3, (.85, .10, .05, .02, .03)),
        ),
        # Illustrative simulator guard, NOT a clinical testing recommendation.
        required_before_benign=("troponin_elevated", "d_dimer_elevated"),
    )


class SyntheticPatient:
    def __init__(self, model: Model, seed: int):
        rng = random.Random(seed)
        self.truth = rng.choices(model.diagnoses, weights=model.priors, k=1)[0]
        index = model.diagnoses.index(self.truth)
        # Pre-sample all outcomes so policies see the same patient, regardless of action order.
        self._outcomes = {a.id: rng.random() < a.positive_likelihood[index] for a in model.actions}
        self._catalog = {a.id: a for a in model.actions}
        self.performed: list[str] = []

    def perform(self, action: Action) -> Observation:
        if self._catalog.get(action.id) != action:
            raise ValueError("Unknown action")
        if action.id in self.performed:
            raise ValueError("Repeated action")
        self.performed.append(action.id)
        return Observation(action.id, self._outcomes[action.id])
