from __future__ import annotations

import os
from typing import Dict, Optional


RUNTIME_ENV_ALLOWLIST = {
    "HOME",
    "PATH",
    "USER",
    "LOGNAME",
    "SHELL",
    "TERM",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "TMP",
    "TEMP",
    "COLORTERM",
    "NO_COLOR",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "VIRTUAL_ENV",
    "PYTHONPATH",
    "XDG_CONFIG_HOME",
    "XDG_CACHE_HOME",
    "XDG_DATA_HOME",
    "XDG_STATE_HOME",
    "CODEX_HOME",
}
RUNTIME_ENV_PREFIX_ALLOWLIST = (
    "ANTHROPIC_",
    "OPENAI_",
    "AWS_",
    "AZURE_OPENAI_",
    "CLAUDE_",
)


def filtered_runtime_env(additional: Optional[dict[str, str]] = None) -> Dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key in RUNTIME_ENV_ALLOWLIST or any(key.startswith(prefix) for prefix in RUNTIME_ENV_PREFIX_ALLOWLIST)
    }
    if additional:
        env.update(additional)
    return env
