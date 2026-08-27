"""Loads settings from a .env file.

A minimal parser rather than python-dotenv: about thirty lines against another
dependency, and the behaviour here is deliberately narrower - flat KEY=VALUE
pairs only, no interpolation.

The real environment takes precedence over the file, so a one-off run can
override a value without editing it.
"""
import os

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(_PROJECT_ROOT, ".env")

_loaded = False


def load_env(path: str = None, override: bool = False) -> dict:
    """Load .env into os.environ; safe to call repeatedly.

    Returns {key: source} for diagnostics, without values, so secrets stay out
    of logs and tracebacks.
    """
    global _loaded
    path = path or ENV_PATH
    applied = {}

    if not os.path.exists(path):
        _loaded = True
        return applied

    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if not key:
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if not value:
                continue  # empty means unset, not an empty string
            if override or not os.environ.get(key):
                os.environ[key] = value
                applied[key] = ".env"
            else:
                applied[key] = "середовище (пріоритет)"

    _loaded = True
    return applied


def describe() -> dict:
    """Configuration state, without revealing secrets."""
    load_env()
    key = os.environ.get("GROQ_API_KEY", "")
    return {
        ".env знайдено": os.path.exists(ENV_PATH),
        "GROQ_API_KEY": f"встановлено ({len(key)} символів)" if key else "НЕ встановлено",
        "VKSK_LLM_PROVIDER": os.environ.get("VKSK_LLM_PROVIDER", "(дефолт: groq)"),
        "VKSK_LLM_MODEL": os.environ.get("VKSK_LLM_MODEL", "(дефолт: openai/gpt-oss-120b)"),
        "VKSK_USE_EMBEDDINGS": os.environ.get("VKSK_USE_EMBEDDINGS", "(вимкнено)"),
    }


# Loaded on import, so variables are in place before anyone reads them.
load_env()


if __name__ == "__main__":
    for k, v in describe().items():
        print(f"  {k}: {v}")
