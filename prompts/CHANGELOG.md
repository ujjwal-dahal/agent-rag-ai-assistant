# Prompt experiment history

These revisions are evidence-driven. The MLflow run descriptions and trace artifacts are the source of the observed failures; this file records the intended decision rule and is updated by the evaluation run with concrete findings.

| Version   | Change driven by the previous trace                                                         | Configuration                                                    |
| --------- | ------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| prompt_v1 | Baseline W16 behavior; its first run establishes the failure trace.                         | `top_k=3`, `chunk_size=300`, `temperature=0`, `max_iterations=3` |
| prompt_v2 | Explicitly refuses empty or irrelevant evidence and limits the agent to one tool selection. | `top_k=2`, `chunk_size=300`, `temperature=0`, `max_iterations=2` |
| prompt_v3 | Prevents repeated identical retries and preserves complete retrieved facts after v2 traces. | `top_k=2`, `chunk_size=450`, `temperature=0`, `max_iterations=2` |
