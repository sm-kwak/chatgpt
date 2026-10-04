import json
import unittest
from unittest.mock import patch
from doctor_agent.agent import Config, DoctorAgent
from doctor_agent.domain import Action, Kind, Model, Observation, information_gain, update
from doctor_agent.experiment import benchmark
from doctor_agent.llm import OpenAICompatibleAdviser
from doctor_agent.simulator import SyntheticPatient, demo_model


class NegativePatient:
    def __init__(self):
        self.calls = []

    def perform(self, action):
        self.calls.append(action.id)
        return Observation(action.id, False)


class AgentTests(unittest.TestCase):
    def test_all_budgets_and_atomic_unique_actions(self):
        model = demo_model()
        for budget in (1, 2, 5, 60):
            for seed in range(100):
                patient = SyntheticPatient(model, seed)
                result = DoctorAgent(model, Config(max_actions=budget)).run(patient)
                self.assertLessEqual(result.attempts, budget)
                self.assertEqual(result.attempts, len(patient.performed))
                self.assertEqual(len(patient.performed), len(set(patient.performed)))
                self.assertAlmostEqual(sum(result.posterior.values()), 1)
                self.assertTrue(all(h["kind"] in {k.value for k in Kind} for h in result.history))

    def test_full_sixty_action_boundary(self):
        actions = tuple(Action(str(i), Kind.QUESTION, "One question", 1, (.5, .5)) for i in range(70))
        model = Model(("a", "b"), (.5, .5), frozenset(), actions)
        patient = NegativePatient()
        result = DoctorAgent(model).run(patient)
        self.assertEqual(result.attempts, 60)
        self.assertEqual(len(patient.calls), 60)
        self.assertEqual(result.reason, "action_budget_exhausted")
        self.assertIsNone(result.diagnosis)

    def test_last_allowed_observation_can_complete_diagnosis(self):
        a = Action("one", Kind.TEST, "One test", 1, (.01, .99))
        model = Model(("a", "b"), (.5, .5), frozenset(), (a,))
        result = DoctorAgent(model, Config(max_actions=1)).run(NegativePatient())
        self.assertEqual(result.status, "diagnosed")
        self.assertEqual(result.diagnosis, "a")

    def test_positive_safety_screen_stops_immediately(self):
        class PositivePatient:
            def perform(self, action):
                return Observation(action.id, True)
        result = DoctorAgent(demo_model()).run(PositivePatient())
        self.assertEqual(result.status, "urgent_evaluation")
        self.assertEqual(result.attempts, 1)
        self.assertIsNone(result.diagnosis)

    def test_failed_or_malformed_environment_consumes_attempt(self):
        class BrokenPatient:
            def perform(self, action):
                raise TimeoutError("private data must not enter trace")
        class MalformedPatient:
            def perform(self, action):
                return Observation(action.id, "false")
        for env in (BrokenPatient(), MalformedPatient()):
            result = DoctorAgent(demo_model()).run(env)
            self.assertEqual(result.attempts, 1)
            self.assertEqual(result.reason, "environment_error")
            self.assertNotIn("private", str(result))

    def test_adviser_cannot_choose_multiple_or_unknown_actions(self):
        class Adviser:
            def choose(self, *args):
                return ["meal_related", "troponin_elevated"]
        patient = NegativePatient()
        result = DoctorAgent(demo_model(), Config(max_actions=3), Adviser()).run(patient)
        self.assertEqual(result.attempts, 3)
        self.assertIn("fallback", result.history[-1]["source"])
        self.assertEqual(patient.calls[:2], ["systolic_bp_low", "severe_dyspnea"])

    def test_benign_diagnosis_blocked_by_remaining_urgent_risk(self):
        model = Model(("benign", "urgent"), (.96, .04), frozenset({"urgent"}),
                      (Action("uninformative", Kind.QUESTION, "One question", 1, (.5, .5)),))
        result = DoctorAgent(model).run(NegativePatient())
        self.assertEqual(result.status, "urgent_evaluation")

    def test_benign_diagnosis_requires_exclusion_checks(self):
        model = Model(("benign", "urgent"), (.999, .001), frozenset({"urgent"}),
                      (Action("exclude", Kind.TEST, "One test", 1, (.5, .5)),), ("exclude",))
        patient = NegativePatient()
        result = DoctorAgent(model).run(patient)
        self.assertEqual(patient.calls, ["exclude"])
        self.assertEqual(result.status, "diagnosed")

    def test_invalid_config(self):
        for budget in (0, 61, True, 2.5):
            with self.assertRaises(ValueError):
                Config(max_actions=budget)

    def test_bayes_and_information_gain(self):
        a = Action("a", Kind.TEST, "One test", 1, (.9, .1))
        self.assertAlmostEqual(update((.5, .5), a, True)[0], .9)
        self.assertGreater(information_gain((.5, .5), a), .5)

    def test_experiments_reproducible(self):
        self.assertEqual(benchmark(30, 123), benchmark(30, 123))


class LLMTests(unittest.TestCase):
    def setUp(self):
        self.action = demo_model().actions[2]

    def mock_response(self, content):
        response = unittest.mock.MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
        return response

    @patch("urllib.request.build_opener")
    def test_valid_response_and_call_cap(self, build):
        build.return_value.open.return_value = self.mock_response('{"action_id":"exertional_pain"}')
        adviser = OpenAICompatibleAdviser("https://example.test/v1", max_calls=1)
        self.assertEqual(adviser.choose({}, [], [self.action]), "exertional_pain")
        self.assertIsNone(adviser.choose({}, [], [self.action]))
        self.assertEqual(build.return_value.open.call_count, 1)
        request = build.return_value.open.call_args.args[0]
        self.assertEqual(json.loads(request.data)["model"], "openai/gpt-oss-20b")

    @patch("urllib.request.build_opener")
    def test_invalid_json_multiple_actions_and_unknown_id(self, build):
        for content in ('not json', '{"action_id":["a","b"]}', '{"action_id":"unknown"}',
                        '{"action_id":"exertional_pain","extra":"x"}'):
            build.return_value.open.return_value = self.mock_response(content)
            adviser = OpenAICompatibleAdviser("https://example.test/v1")
            self.assertIsNone(adviser.choose({}, [], [self.action]))
            self.assertEqual(adviser.failures, 1)

    @patch("urllib.request.build_opener")
    def test_timeout_falls_back_without_retry(self, build):
        build.return_value.open.side_effect = TimeoutError("sensitive")
        adviser = OpenAICompatibleAdviser("https://example.test/v1")
        self.assertIsNone(adviser.choose({}, [], [self.action]))
        self.assertEqual(build.return_value.open.call_count, 1)

    def test_url_validation(self):
        for url in ("http://remote.example/v1", "https://key@example.test/v1", "https://example.test/v1?key=secret"):
            with self.assertRaises(ValueError):
                OpenAICompatibleAdviser(url)


if __name__ == "__main__":
    unittest.main()
