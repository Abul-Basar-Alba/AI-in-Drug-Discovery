#!/usr/bin/env bash
# Setup a local Python virtual environment and install requirements
set -euo pipefail

PYTHON=python3
VENV_DIR=.venv

echo "Creating virtual environment in ${VENV_DIR} using ${PYTHON}..."
$PYTHON -m venv ${VENV_DIR}
echo "Installing pip and wheel..."
${VENV_DIR}/bin/python -m pip install --upgrade pip wheel
echo "Installing requirements from requirements.txt (this may take a few minutes)..."
set +e
${VENV_DIR}/bin/pip install -r requirements.txt
RC=$?
set -e
if [ $RC -ne 0 ]; then
	echo "Warning: pip reported errors while installing some packages." >&2
	echo "Common issues: platform-specific packages (RDKit) or mismatched Python version." >&2
	echo "You can retry installing problematic packages manually, or install RDKit via conda as documented in README.md." >&2
fi
if [ $RC -ne 0 ]; then
	echo "Attempting to install a minimal set of packages (requirements-min.txt) to run the core pipeline..."
	${VENV_DIR}/bin/pip install -r requirements-min.txt || {
		echo "Failed to install minimal requirements as well. Please inspect the errors above and consider using a Python 3.10/3.11 environment or conda." >&2
		exit 1
	}
	echo "Minimal requirements installed. You can still try installing full requirements manually later." >&2
fi

echo "Done. Activate the environment with:"
echo "  source ${VENV_DIR}/bin/activate"
