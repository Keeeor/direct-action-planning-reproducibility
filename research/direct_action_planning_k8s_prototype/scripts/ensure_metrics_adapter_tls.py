from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile


def _run(argv: list[str], *, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, input=input_text, text=True, capture_output=True, check=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create the local TLS Secret used by the prototype Metrics API adapter."
    )
    parser.add_argument("--context", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--secret", default="dap-prototype-metrics-adapter-tls")
    parser.add_argument("--service", default="dap-prototype-metrics-server")
    parser.add_argument("--rotate", action="store_true")
    args = parser.parse_args()
    if shutil.which("openssl") is None:
        raise SystemExit("openssl is required to generate the local adapter TLS certificate")

    base = ["kubectl", "--context", args.context, "-n", args.namespace]
    exists = _run(base + ["get", "secret", args.secret])
    if exists.returncode == 0 and not args.rotate:
        print(f"reusing existing TLS Secret {args.namespace}/{args.secret}")
        return 0
    with tempfile.TemporaryDirectory(prefix="dap-metrics-tls-") as temporary:
        directory = Path(temporary)
        certificate = directory / "tls.crt"
        key = directory / "tls.key"
        subject = f"/CN={args.service}.{args.namespace}.svc"
        generated = _run(
            [
                "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "30",
                "-keyout", str(key), "-out", str(certificate), "-subj", subject,
            ]
        )
        if generated.returncode != 0:
            raise SystemExit(f"openssl certificate generation failed: {generated.stderr}")
        rendered = _run(
            base
            + [
                "create", "secret", "tls", args.secret, "--cert", str(certificate), "--key", str(key),
                "--dry-run=client", "-o", "yaml",
            ]
        )
        if rendered.returncode != 0:
            raise SystemExit(f"kubectl secret render failed: {rendered.stderr}")
        applied = _run(base + ["apply", "-f", "-"], input_text=rendered.stdout)
        if applied.returncode != 0:
            raise SystemExit(f"kubectl secret apply failed: {applied.stderr}")
    print(applied.stdout.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
