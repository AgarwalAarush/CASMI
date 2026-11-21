"""Build a code-only ZIP for attachment to an offline Kaggle notebook."""

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


SOURCE_FILES = ("casmi_run.py", "casmi_baseline.py", "casmi_catalog.py", "requirements.txt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/casmi-code.zip"))
    output = parser.parse_args().output
    root = Path(__file__).resolve().parents[1]
    files = {name: (root / name).read_bytes() for name in SOURCE_FILES}
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w") as archive:
        for name, contents in files.items():
            archive.writestr(ZipInfo(name), contents, compress_type=ZIP_DEFLATED)
    print(output)


if __name__ == "__main__":
    main()
