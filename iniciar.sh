#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
exec /home/jonas/.venv-ml/bin/python iniciar_demo.py "$@"

