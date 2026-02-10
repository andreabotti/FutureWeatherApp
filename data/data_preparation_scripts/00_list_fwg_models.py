from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path


JAVA_SRC = r"""
import futureweathergenerator_europe.ModelType;

public class ListFWGModels {
    public static void main(String[] args) {
        for (ModelType m : ModelType.values()) {
            System.out.println(m.name());
        }
    }
}
"""


def main() -> None:
    p = argparse.ArgumentParser(description="List FWG Europe v2.0.2 ModelType tokens programmatically from the jar.")
    p.add_argument("--jar", required=True, help="Path to FutureWeatherGenerator_Europe_v2.0.2.jar")
    p.add_argument("--out", default="models.txt", help="Output text file (default: models.txt)")
    p.add_argument("--javac", default="javac", help="javac executable (default: javac)")
    p.add_argument("--java", default="java", help="java executable (default: java)")
    args = p.parse_args()

    jar_path = Path(args.jar).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    if not jar_path.exists():
        raise FileNotFoundError(f"Jar not found: {jar_path}")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        src = td_path / "ListFWGModels.java"
        src.write_text(JAVA_SRC, encoding="utf-8")

        # Compile, with the FWG jar on the classpath so ModelType can be resolved
        compile_cmd = [args.javac, "-cp", str(jar_path), str(src)]
        subprocess.run(compile_cmd, check=True)

        # Run, again with FWG jar on classpath
        run_cmd = [args.java, "-cp", f"{jar_path}{';' if True else ':'}{td_path}", "ListFWGModels"]
        # Note: Windows uses ';' as classpath separator. The conditional is kept simple.
        result = subprocess.run(run_cmd, check=True, capture_output=True, text=True)

        models = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        models_sorted = sorted(set(models))

        out_path.write_text("\n".join(models_sorted) + "\n", encoding="utf-8")

    print(f"Wrote {len(models_sorted)} model tokens to: {out_path}")


if __name__ == "__main__":
    main()
