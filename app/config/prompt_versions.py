from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PromptConfig:
    version: str
    temperature: float
    top_k: int
    chunk_size: int
    max_iterations: int
    prompt_path: Path

    @property
    def prompt(self) -> str:
        return self.prompt_path.read_text(encoding="utf-8")


PROMPT_CONFIGS = {
    "prompt_v1": PromptConfig("prompt_v1", 0.0, 3, 300, 3, ROOT / "prompts" / "v1.txt"),
    "prompt_v2": PromptConfig("prompt_v2", 0.0, 2, 300, 2, ROOT / "prompts" / "v2.txt"),
    "prompt_v3": PromptConfig("prompt_v3", 0.0, 2, 450, 2, ROOT / "prompts" / "v3.txt"),
}


def get_prompt_config(version: str) -> PromptConfig:
    try:
        return PROMPT_CONFIGS[version]
    except KeyError as exc:
        raise ValueError(f"Unknown prompt version: {version}") from exc