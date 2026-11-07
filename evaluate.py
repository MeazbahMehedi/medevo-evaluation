import argparse
import json
from pathlib import Path
import sys

from src.runtime import read_settings, safe_error


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Prepare medical VQA data and record reproducible baseline evaluations")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "benchmarks")
    parser.add_argument("--settings", type=Path, default=ROOT / "settings.json")
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs" / "benchmark_protocol.json")
    parser.add_argument("--worker-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--repair-worker-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--collection-worker-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--readout-worker-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--audit-worker-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--pilot-config", type=Path, default=ROOT / "configs" / "repair_pilot.json")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("prepare", help="Download pinned sources and prepare separated data manifests")
    commands.add_parser("inspect", help="Validate the prepared manifests and image files")
    run = commands.add_parser("run", help="Generate baseline answers; feedback role by default")
    run.add_argument("--benchmark", required=True, choices=("vqa_rad", "slake"))
    run.add_argument("--role", choices=("feedback", "test"), default="feedback")
    size = run.add_mutually_exclusive_group()
    size.add_argument("--limit", type=int, default=5)
    size.add_argument("--full", action="store_true")
    run.add_argument("--final", action="store_true", help="Explicitly allow test-set generation after protocol freeze")
    run.add_argument("--resume", type=Path)
    score = commands.add_parser("score", help="Score stored predictions without rerunning the model")
    score.add_argument("--run-dir", type=Path, required=True)
    compare = commands.add_parser("compare-prompts", help="Compare a concise prompt with two completed feedback runs")
    compare.add_argument("--vqa-rad-run", type=Path, required=True)
    compare.add_argument("--slake-run", type=Path, required=True)
    diagnose = commands.add_parser("diagnose-answers", help="Compare boxed and whole-answer scores without new generation")
    diagnose.add_argument("--comparison-dir", type=Path, required=True)
    commands.add_parser("repair-pilot", help="Run three training-only reasoning/audit/continuation cases without adapter training")
    review = commands.add_parser("inspect-repair", help="Revalidate saved auditor decisions without rerunning the pilot")
    review.add_argument("--run-dir", type=Path, required=True)
    collect = commands.add_parser("collect-training", help="Collect twelve training cases, reusing the three saved non-repair cases")
    collect.add_argument("--previous-pilot", type=Path, required=True)
    readout = commands.add_parser("compare-readouts", help="Compare twelve saved reasoning readouts with a text-only extraction method")
    readout.add_argument("--collection-dir", type=Path, required=True)
    audit = commands.add_parser("compare-audits", help="Compare twelve saved audits with a shorter system instruction")
    audit.add_argument("--collection-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.command and not any((args.worker_dir, args.repair_worker_dir, args.collection_worker_dir,
                                    args.readout_worker_dir, args.audit_worker_dir)):
        parser.print_help()
        return 0
    try:
        from src.benchmarks import load_bundle, prepare_bundle
        from src.evaluation import create_run, evaluation_worker, execute_run, read_protocol, score_run

        data_dir = args.data_dir.resolve()
        if args.audit_worker_dir:
            from src.audit_experiment import audit_worker
            return audit_worker(ROOT, args.audit_worker_dir.resolve())
        if args.readout_worker_dir:
            from src.readout_experiment import readout_worker
            return readout_worker(ROOT, args.readout_worker_dir.resolve())
        if args.collection_worker_dir:
            from src.training_collection import collection_worker
            return collection_worker(ROOT, data_dir, args.collection_worker_dir.resolve())
        if args.repair_worker_dir:
            from src.repair_pilot import pilot_worker
            return pilot_worker(ROOT, data_dir, args.repair_worker_dir.resolve())
        if args.worker_dir:
            return evaluation_worker(args.worker_dir.resolve(), data_dir)
        if args.command == "prepare":
            sources = json.loads((ROOT / "configs" / "data_sources.json").read_text(encoding="utf-8"))
            result = prepare_bundle(data_dir, sources, read_protocol(args.protocol))
        elif args.command == "inspect":
            result, _ = load_bundle(data_dir, verify_images=True)
        elif args.command == "score":
            result = score_run(args.run_dir.resolve(), data_dir)
        elif args.command == "compare-prompts":
            from src.prompt_experiment import compare_prompts
            result = compare_prompts(ROOT, data_dir,
                       {"vqa_rad": args.vqa_rad_run.resolve(), "slake": args.slake_run.resolve()},
                       read_settings(args.settings), read_protocol(args.protocol))
            print(json.dumps(result, indent=2))
            return 0 if result["status"] == "complete" else 2
        elif args.command == "diagnose-answers":
            from src.answer_diagnostic import diagnose_answers
            result = diagnose_answers(ROOT, data_dir, args.comparison_dir.resolve())
            print(json.dumps(result, indent=2))
            return 0 if result["status"] == "complete" else 2
        elif args.command == "repair-pilot":
            from src.repair_pilot import read_pilot_config, start_repair_pilot
            result = start_repair_pilot(ROOT, data_dir, read_settings(args.settings), read_pilot_config(args.pilot_config))
            print(json.dumps(result, indent=2))
            return 0 if result["status"] == "complete" else 2
        elif args.command == "inspect-repair":
            from src.repair_review import inspect_repair
            result = inspect_repair(ROOT, args.run_dir.resolve())
            print(json.dumps(result, indent=2))
            return 0
        elif args.command == "compare-audits":
            from src.audit_experiment import start_audit_experiment
            from src.repair_pilot import read_pilot_config
            result = start_audit_experiment(ROOT, data_dir, args.collection_dir.resolve(),
                                            read_settings(args.settings), read_pilot_config(args.pilot_config))
            print(json.dumps(result, indent=2))
            return 0 if result["status"] == "complete" else 2
        elif args.command == "compare-readouts":
            from src.readout_experiment import start_readout_experiment
            result = start_readout_experiment(ROOT, data_dir, args.collection_dir.resolve(), read_settings(args.settings))
            print(json.dumps(result, indent=2))
            return 0 if result["status"] == "complete" else 2
        elif args.command == "collect-training":
            from src.repair_pilot import read_pilot_config
            from src.training_collection import start_collection
            result = start_collection(ROOT, data_dir, args.previous_pilot.resolve(),
                                      read_settings(args.settings), read_pilot_config(args.pilot_config))
            print(json.dumps(result, indent=2))
            return 0 if result["status"] == "complete" else 2
        else:
            if args.role == "test" and not args.final:
                raise ValueError("Test generation is reserved for the frozen final protocol; use --final only then")
            settings, protocol = read_settings(args.settings), read_protocol(args.protocol)
            import psutil
            if psutil.virtual_memory().available < settings["minimum_available_gib"] * 1024**3:
                raise RuntimeError("Available RAM is below the configured minimum")
            directory = create_run(ROOT, data_dir, args.benchmark, args.role,
                                   None if args.full else args.limit, settings, protocol, args.resume)
            print(f"Output folder: {directory}", flush=True)
            result = execute_run(ROOT, directory, data_dir, settings, protocol)
            resources = result["worker"]
            print(json.dumps(result, indent=2))
            return 0 if result["summary"]["status"] == "complete" and resources["worker_exit_code"] == 0 and not resources["limit_reached"] else 2
        print(json.dumps(result, indent=2))
        return 0 if args.command != "score" or result["summary"]["status"] == "complete" else 2
    except (Exception, KeyboardInterrupt) as error:
        print(json.dumps({"status": "failed", "error": safe_error(error)}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
