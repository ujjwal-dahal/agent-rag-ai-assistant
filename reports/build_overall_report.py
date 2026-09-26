"""Build and log the static comparison dashboard for all prompt versions."""

import argparse
import html
import json
from pathlib import Path

import mlflow


ROOT = Path(__file__).resolve().parents[1]
COMPARISON_PATH = ROOT / "mlruns_artifacts" / "comparison.json"
ARTIFACT_ROOT = ROOT / "mlruns_artifacts"
OUTPUT_PATH = ROOT / "reports" / "overall_comparison_report.html"


def load_data() -> tuple[list[dict], dict[str, dict]]:
    comparison = json.loads(COMPARISON_PATH.read_text(encoding="utf-8"))
    reports = {
        row["version"]: json.loads(
            (ARTIFACT_ROOT / row["version"] / "evidently_report.json").read_text(encoding="utf-8")
        )
        for row in comparison
    }
    return comparison, reports


def choose_promotion(comparison: list[dict], reports: dict[str, dict]) -> tuple[dict, str]:
    passing = [
        row for row in comparison
        if reports[row["version"]].get("summary", {}).get("all_passed", False)
    ]
    candidates = passing or comparison
    winner = max(candidates, key=lambda row: (row["pct_tests_passed"], -row["avg_tokens"]))
    reason = (
        "Promoted because it has the highest Evidently regression pass rate among passing suites; "
        "token cost breaks any tie."
        if passing
        else "No suite passed all checks, so this is the highest regression score available; review before promotion."
    )
    return winner, reason


def failure_patterns(report: dict) -> list[str]:
    patterns = []
    for check_name, cases in report.get("details", {}).items():
        for case in cases:
            verdict = case.get("verdict", {})
            if verdict.get("category") != "correct":
                patterns.append(
                    f"{check_name}: {case.get('id', 'unknown')} -> "
                    f"{verdict.get('category', 'unknown')}"
                )
    return patterns or ["No incorrect or unknown judge verdicts."]


def render(comparison: list[dict], reports: dict[str, dict], winner: dict, reason: str) -> str:
    rows = []
    cards = []
    for row in comparison:
        version = row["version"]
        report = reports[version]
        promoted = version == winner["version"]
        report_path = f"../mlruns_artifacts/{version}/evidently_report.html"
        rows.append(
            "<tr class=\"{}\"><td><strong>{}</strong>{}</td><td>{:.1%}</td><td>{:.2f}</td>"
            "<td>{:,.2f}</td><td>{:.1%}</td></tr>".format(
                "winner" if promoted else "",
                html.escape(version),
                " <span class=\"badge\">PROMOTED</span>" if promoted else "",
                row["task_success_rate"],
                row["avg_iterations"],
                row["avg_tokens"],
                row["pct_tests_passed"],
            )
        )
        patterns = "".join(f"<li>{html.escape(pattern)}</li>" for pattern in failure_patterns(report))
        cards.append(
            f"<article><div class=\"card-title\"><h3>{html.escape(version)}</h3>"
            f"<a href=\"{report_path}\" target=\"_blank\">Open Evidently report</a></div>"
            f"<p><strong>Failure patterns</strong></p><ul>{patterns}</ul>"
            f"<iframe title=\"{html.escape(version)} Evidently report\" src=\"{report_path}\"></iframe></article>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agentic AI Prompt Version Comparison</title>
<style>
:root {{ color-scheme: light; --ink:#17202a; --muted:#607080; --line:#d9e1e8; --accent:#0b6e69; --win:#e6f5ee; --bg:#f7f9fb; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 Georgia,serif; }}
main {{ max-width:1180px; margin:0 auto; padding:44px 24px 64px; }} h1,h2,h3 {{ font-family:Georgia,serif; letter-spacing:0; }} h1 {{ font-size:36px; margin:0 0 8px; }}
.dek {{ color:var(--muted); max-width:760px; margin:0 0 28px; }} section {{ background:white; border:1px solid var(--line); padding:24px; margin:18px 0; }}
.promoted {{ border-left:5px solid var(--accent); background:var(--win); }} .promoted h2 {{ margin:0 0 6px; }}
table {{ width:100%; border-collapse:collapse; }} th,td {{ text-align:left; padding:13px 12px; border-bottom:1px solid var(--line); }} th {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }} tr.winner {{ background:var(--win); }}
.badge {{ color:white; background:var(--accent); font:11px Arial,sans-serif; padding:4px 7px; margin-left:8px; }} .cards {{ display:grid; grid-template-columns:repeat(3,1fr); gap:16px; }} article {{ border:1px solid var(--line); padding:16px; min-width:0; }}
.card-title {{ display:flex; justify-content:space-between; gap:10px; align-items:baseline; }} article h3 {{ margin:0; }} a {{ color:var(--accent); }} iframe {{ width:100%; height:360px; border:1px solid var(--line); margin-top:12px; }} li {{ margin:5px 0; color:var(--muted); }}
@media (max-width:800px) {{ main {{ padding:28px 14px; }} h1 {{ font-size:29px; }} section {{ padding:16px; overflow-x:auto; }} table {{ min-width:680px; }} .cards {{ grid-template-columns:1fr; }} iframe {{ height:300px; }} }}
</style></head><body><main>
<h1>Agentic AI prompt comparison</h1><p class="dek">A single comparison of the real W17 runs for prompt_v1, prompt_v2, and prompt_v3. Source data: <code>mlruns_artifacts/comparison.json</code> and each version's Evidently JSON/HTML report.</p>
<section class="promoted"><h2>Promoted: {html.escape(winner['version'])}</h2><p>{html.escape(reason)}</p></section>
<section><h2>Metrics at a glance</h2><table><thead><tr><th>Version</th><th>Task success</th><th>Avg iterations</th><th>Avg tokens</th><th>Evidently pass rate</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>
<section><h2>Individual Evidently reports</h2><div class="cards">{''.join(cards)}</div></section>
<section><h2>Interpretation</h2><p>v2 is the selected version because it passes 7/8 judge cases (87.5%) while retaining 100% task success. v3 uses the fewest average tokens but fails the regression gate: its report records an empty <code>missing_knowledge</code> answer and unnecessary tool selection in the <code>no_tool</code> case. v1 also fails the <code>no_tool</code> correctness and tool-choice checks.</p></section>
</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default=f"sqlite:///{ROOT / 'mlflow.db'}")
    args = parser.parse_args()
    comparison, reports = load_data()
    winner, reason = choose_promotion(comparison, reports)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(render(comparison, reports, winner, reason), encoding="utf-8")
    mlflow.set_tracking_uri(args.tracking_uri)
    with mlflow.start_run(run_name="overall-comparison", description="Combined Evidently and MLflow prompt-version dashboard"):
        mlflow.log_param("promoted_version", winner["version"])
        mlflow.log_param("promotion_reason", reason)
        for row in comparison:
            prefix = row["version"]
            mlflow.log_metrics({
                f"{prefix}_task_success_rate": row["task_success_rate"],
                f"{prefix}_avg_iterations": row["avg_iterations"],
                f"{prefix}_avg_tokens": row["avg_tokens"],
                f"{prefix}_pct_tests_passed": row["pct_tests_passed"],
            })
        mlflow.log_artifact(str(OUTPUT_PATH), artifact_path="overall_comparison")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()