import argparse
from dataclasses import asdict
import json
from pathlib import Path
from .agent import Config, DoctorAgent
from .experiment import benchmark
from .llm import OpenAICompatibleAdviser
from .simulator import SyntheticPatient, demo_model


def main():
    parser = argparse.ArgumentParser(description="Synthetic diagnostic agent lab — not for clinical use")
    commands = parser.add_subparsers(dest="command", required=True)
    episode = commands.add_parser("episode")
    episode.add_argument("--seed", type=int, default=42)
    episode.add_argument("--max-actions", type=int, default=60)
    episode.add_argument("--llm", action="store_true", help="Explicitly enable API calls for synthetic observations")
    evaluation = commands.add_parser("benchmark")
    evaluation.add_argument("--cases", type=int, default=1000)
    evaluation.add_argument("--seed", type=int, default=10000)
    evaluation.add_argument("--max-actions", type=int, default=60)
    evaluation.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "benchmark":
            result = benchmark(args.cases, args.seed, args.max_actions)
        else:
            model = demo_model()
            adviser = OpenAICompatibleAdviser.from_environment() if args.llm else None
            patient = SyntheticPatient(model, args.seed)
            result = asdict(DoctorAgent(model, Config(max_actions=args.max_actions), adviser).run(patient))
            result["simulation_truth_for_evaluation_only"] = patient.truth
            if adviser:
                result["llm"] = {"model": adviser.model, "calls": adviser.calls, "failures": adviser.failures,
                                 "last_error": adviser.last_error}
        output = json.dumps(result, indent=2, ensure_ascii=False)
        if args.command == "benchmark" and args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output + "\n", encoding="utf-8")
        print(output)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
