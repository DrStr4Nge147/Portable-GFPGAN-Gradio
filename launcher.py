import os
import sys
import subprocess
import platform
from pathlib import Path

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
APP_DIR = SCRIPT_DIR / "app"
SYSTEM_DIR = SCRIPT_DIR / "system"
PYTHON_EXE = SYSTEM_DIR / "python" / "python.exe"
REQS_FILE = APP_DIR / "requirements.txt"
APP_SCRIPT = APP_DIR / "app.py"

# --- Cache Directories for downloads ---
CACHE_DIR = SCRIPT_DIR / "cache"
PIP_CACHE_DIR = CACHE_DIR / "pip_cache"
TORCH_HOME = CACHE_DIR / "torch_cache"

# --- Marker Files ---
REQS_MARKER = CACHE_DIR / "requirements_installed.flag"
PYTORCH_MARKER = CACHE_DIR / "pytorch_installed.flag"
FILTERPY_MARKER = CACHE_DIR / "filterpy_installed.flag"

# --- Default PyTorch Versions ---
DEFAULT_TORCH_VERSION = "2.0.1"
DEFAULT_TORCHVISION_VERSION = "0.15.2"
DEFAULT_TORCHAUDIO_VERSION = "2.0.2"

# Will be overridden by args
TARGET_CUDA_VERSION_SUFFIX = ""
EXTRA_INDEX_URL = ""


def run_command(command, description, check=True, capture_output=False, env=None):
    print(f"--- Running: {description} ---")
    cmd = [str(c) for c in command]
    print("Executing:", ' '.join(cmd))
    try:
        env_vars = os.environ.copy()
        if env:
            env_vars.update(env)
        result = subprocess.run(
            cmd, check=check, capture_output=capture_output,
            text=True, env=env_vars, encoding='utf-8', errors='replace'
        )
        if capture_output:
            print(result.stdout)
            if result.stderr:
                print("--- Stderr ---", file=sys.stderr)
                print(result.stderr, file=sys.stderr)
                print("--- End Stderr ---", file=sys.stderr)
        print(f"--- Completed: {description} ---")
        return result
    except subprocess.CalledProcessError as e:
        print(f"!!! ERROR during: {description} !!!", file=sys.stderr)
        print(f"Exit code {e.returncode}", file=sys.stderr)
        if e.stdout:
            print(e.stdout, file=sys.stderr)
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        if check:
            raise
        return None
    except KeyboardInterrupt:
        # Graceful interrupt
        print("\nKeyboardInterrupt during: {description}. Exiting gracefully.")
        sys.exit(0)


def setup_environment():
    """Creates local cache dirs and sets pip/torch cache env vars."""
    print("--- Setting up cache directories ---")
    CACHE_DIR.mkdir(exist_ok=True)
    PIP_CACHE_DIR.mkdir(exist_ok=True)
    TORCH_HOME.mkdir(exist_ok=True)

    os.environ['PIP_CACHE_DIR'] = str(PIP_CACHE_DIR)
    os.environ['TORCH_HOME'] = str(TORCH_HOME)

    print(f"PIP_CACHE_DIR set to {os.environ['PIP_CACHE_DIR']}")
    print(f"TORCH_HOME set to {os.environ['TORCH_HOME']}")
    print("--- Cache setup complete ---")


def check_pytorch(target_version, cuda_suffix):
    if PYTORCH_MARKER.exists():
        print("Found PyTorch marker, verifying...")
        res = run_command([
            str(PYTHON_EXE), "-c", "import torch; print(torch.__version__); print(torch.cuda.is_available())"
        ], "Verify existing PyTorch", check=False, capture_output=True)
        if res and res.returncode == 0:
            print("PyTorch verification successful.")
            return True
        PYTORCH_MARKER.unlink(missing_ok=True)
        print("Marker invalid, reinstalling.")

    print(f"Checking for PyTorch {target_version}{cuda_suffix}...")
    res = run_command([
        str(PYTHON_EXE), "-c", "import torch; print(torch.__version__)"
    ], "Check PyTorch import", check=False, capture_output=True)
    if not res or res.returncode != 0:
        print("PyTorch import failed.")
        return False

    installed = res.stdout.strip().split('+')[0]
    if installed != target_version:
        print(f"Version mismatch: {installed} != {target_version}")
        return False

    print("Compatible PyTorch found.")
    PYTORCH_MARKER.touch()
    return True


def install_pytorch(version, vision, audio, cuda_suffix, index_url):
    print("--- Installing PyTorch and deps ---")
    if PYTORCH_MARKER.exists():
        PYTORCH_MARKER.unlink(missing_ok=True)

    pkgs = [
        f"torch=={version}{cuda_suffix}",
        f"torchvision=={vision}{cuda_suffix}",
        f"torchaudio=={audio}{cuda_suffix}"
    ]
    cmd = [str(PYTHON_EXE), "-m", "pip", "install", "--no-cache-dir", "--force-reinstall"] + pkgs
    if index_url:
        cmd += ["--extra-index-url", index_url]

    run_command(cmd, "Install PyTorch packages")
    if check_pytorch(version, cuda_suffix):
        print("PyTorch installed and verified.")
        return True
    print("PyTorch install or verification failed.")
    return False


def install_filterpy():
    if FILTERPY_MARKER.exists():
        print("filterpy already installed.")
        return True

    run_command([
        str(PYTHON_EXE), "-m", "pip", "install", "filterpy", "--no-cache-dir", "--use-pep517"
    ], "Install filterpy")
    try:
        run_command([str(PYTHON_EXE), "-c", "import filterpy"], "Verify filterpy", check=True)
        FILTERPY_MARKER.touch()
        return True
    except:
        return False


def install_requirements():
    if REQS_MARKER.exists():
        print("Requirements already installed.")
        return True
    if not REQS_FILE.is_file():
        print(f"Missing requirements.txt at {REQS_FILE}")
        return False

    run_command([
        str(PYTHON_EXE), "-m", "pip", "install", "-r", str(REQS_FILE), "--no-cache-dir"
    ], "Install requirements")
    REQS_MARKER.touch()
    return True


def run_application():
    print("Launching GFPGAN application...")
    try:
        run_command([str(PYTHON_EXE), str(APP_SCRIPT)], "Run application", check=False)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Shutting down application gracefully.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-suffix", default="")
    parser.add_argument("--torch-version", default=DEFAULT_TORCH_VERSION)
    parser.add_argument("--torchvision-version", default=DEFAULT_TORCHVISION_VERSION)
    parser.add_argument("--torchaudio-version", default=DEFAULT_TORCHAUDIO_VERSION)
    parser.add_argument("--index-url", default="")
    args = parser.parse_args()

    TARGET_CUDA_VERSION_SUFFIX = args.cuda_suffix
    EXTRA_INDEX_URL = args.index_url

    if not PYTHON_EXE.is_file():
        print(f"Python not found at {PYTHON_EXE}", file=sys.stderr)
        sys.exit(1)

    setup_environment()

    try:
        run_command([str(PYTHON_EXE), "-m", "pip", "install", "--upgrade", "pip", "--no-cache-dir"], "Upgrade pip")
    except:
        print("Warning: pip upgrade failed, continuing...")

    if not check_pytorch(args.torch_version, TARGET_CUDA_VERSION_SUFFIX):
        if not install_pytorch(args.torch_version, args.torchvision_version,
                               args.torchaudio_version, TARGET_CUDA_VERSION_SUFFIX,
                               EXTRA_INDEX_URL):
            print("Failed to install PyTorch", file=sys.stderr)
            sys.exit(1)

    if not install_filterpy():
        print("Failed to install filterpy", file=sys.stderr)
        sys.exit(1)

    if not install_requirements():
        print("Failed to install requirements", file=sys.stderr)
        sys.exit(1)

    try:
        run_application()
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received in launcher. Exiting.")
        sys.exit(0)