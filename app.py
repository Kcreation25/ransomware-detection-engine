"""
app.py
======
Thin CLI dispatcher so the project has one obvious entry point.

    python app.py train
    python app.py predict path/to/wallets.csv
    python app.py dashboard      # launches the Streamlit app

For the actual Streamlit UI, see streamlit_app/main.py — Streamlit apps
need to be launched via `streamlit run`, so this just shells out to that.
"""

from __future__ import annotations

import subprocess
import sys


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    command = sys.argv[1]

    if command == "train":
        import train
        train.main()
    elif command == "predict":
        sys.argv = ["predict.py"] + sys.argv[2:]
        import predict
        predict.main()
    elif command == "dashboard":
        subprocess.run(["streamlit", "run", "streamlit_app/main.py"])
    else:
        print(f"Unknown command: {command}\n")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
