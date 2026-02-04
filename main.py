import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fill a DOCX form from JSON using a local LLM.")
    parser.add_argument("--input-docx", "--template", required=True, help="Path to the input DOCX template.")
    parser.add_argument("--input-json", "--data", required=True, help="Path to the JSON data file.")
    parser.add_argument("--output-docx", "--out", default=None, help="Output DOCX path.")
    parser.add_argument("--report", default=None, help="Optional report.json output path.")
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Hugging Face model name for mapping labels to JSON keys.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Directory for debug outputs and reports.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug artifacts (default on).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the pipeline without writing the output DOCX.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero code if unmatched anchors remain.",
    )
    return parser.parse_args()


def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def main() -> None:
    project_root = Path(__file__).resolve().parent
    _load_env_file(project_root / ".env")
    os.environ.setdefault("HF_HOME", str(project_root / ".hf_cache"))
    if "HUGGINGFACEHUB_API_TOKEN" in os.environ and "HF_TOKEN" not in os.environ:
        os.environ["HF_TOKEN"] = os.environ["HUGGINGFACEHUB_API_TOKEN"]
    if "HF_TOKEN" in os.environ and "HUGGINGFACE_HUB_TOKEN" not in os.environ:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
    if "HF_TOKEN" in os.environ and "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ["HF_TOKEN"]

    from src.pipeline.graph import run_pipeline

    args = parse_args()
    input_docx = Path(args.input_docx)
    input_json = Path(args.input_json)
    output_docx = Path(args.output_docx) if args.output_docx else input_docx.with_suffix(".filled.docx")

    result = run_pipeline(
        input_docx=input_docx,
        input_json=input_json,
        output_docx=output_docx,
        model_name=args.model_name,
        artifacts_dir=Path(args.artifacts_dir),
        dry_run=args.dry_run,
        strict=args.strict,
    )

    if args.report:
        from src.report.make_report import write_report

        write_report(Path(args.report), result.get("report", {}))


if __name__ == "__main__":
    main()
