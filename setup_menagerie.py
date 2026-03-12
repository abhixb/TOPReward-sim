"""
Downloads the SO-ARM100 model from MuJoCo Menagerie.
Run once: python setup_menagerie.py
"""

import pathlib
import subprocess
import sys


def main():
    dest = pathlib.Path("mujoco_menagerie")

    if (dest / "trs_so_arm100").exists():
        print("SO-ARM100 model already present. Skipping.")
        return

    # Try main MuJoCo Menagerie repo first
    print("Cloning MuJoCo Menagerie (sparse, SO-ARM100 only)...")
    if dest.exists():
        print(f"Removing existing {dest} directory...")
        subprocess.run(["rm", "-rf", str(dest)], check=True)

    try:
        subprocess.run(
            [
                "git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                "https://github.com/google-deepmind/mujoco_menagerie.git",
                str(dest),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "sparse-checkout", "set", "trs_so_arm100"],
            cwd=str(dest),
            check=True,
        )

        if (dest / "trs_so_arm100").exists() and any((dest / "trs_so_arm100").iterdir()):
            print("Done! Model at:", dest / "trs_so_arm100")
            return
        else:
            print("trs_so_arm100 not found in main repo. Trying fork...")
            subprocess.run(["rm", "-rf", str(dest)], check=True)
    except subprocess.CalledProcessError:
        print("Failed to clone main repo. Trying fork...")
        if dest.exists():
            subprocess.run(["rm", "-rf", str(dest)], check=True)

    # Fall back to fork with the SO-ARM100 model
    print("Cloning from chernyadev/mujoco_menagerie fork (branch: add-so-arm100)...")
    try:
        subprocess.run(
            [
                "git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                "--branch", "add-so-arm100",
                "https://github.com/chernyadev/mujoco_menagerie.git",
                str(dest),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "sparse-checkout", "set", "trs_so_arm100"],
            cwd=str(dest),
            check=True,
        )

        if (dest / "trs_so_arm100").exists() and any((dest / "trs_so_arm100").iterdir()):
            print("Done! Model at:", dest / "trs_so_arm100")
        else:
            print("ERROR: trs_so_arm100 not found even in fork.", file=sys.stderr)
            sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to clone fork: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
