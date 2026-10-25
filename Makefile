BLENDER ?= /Applications/Blender.app/Contents/MacOS/Blender
PYTHON ?= $(if $(wildcard .venv/bin/python),$(abspath .venv/bin/python),python3)
RUFF ?= $(if $(wildcard .venv/bin/ruff),$(abspath .venv/bin/ruff),ruff)
JOBS ?= 10
RUNTIME ?= ../sprongmotion-runtime
MODEL_DIR ?=
RT = $(RUNTIME)/build/dist/sprongmotion-1.1.0
LIBS ?= macos-arm64=$(RT)-macos-arm64/lib/libsprongmotion.dylib linux-x64=$(RT)-linux-x86_64/lib/libsprongmotion.so \
	windows-x64=$(RT)-windows-x86_64/lib/libsprongmotion.dll
SHELL := /bin/bash -o pipefail
HAS_PYTEST = $(shell $(PYTHON) -c "import pytest, numpy" 2>/dev/null && echo yes)
NO_PYTEST = echo "$(PYTHON) has no pytest and numpy: pip install pytest numpy, or pass PYTHON=/path/to/python"
run_unit = cd tests/unit && $(PYTHON) -m pytest -q -p no:cacheprovider --rootdir . .
FIRST = tests/test_stability.py tests/test_paths.py tests/test_paths_edits.py tests/test_blocks.py  # the longest files start first
TESTS = $(FIRST) $(filter-out $(FIRST),$(sort $(wildcard tests/test_*.py)))
# each Blender test file in its own Blender, side by side, two model threads each; $(1): extra environment
blender_tests = printf '%s\n' $(TESTS) | env SM_THREADS=2 $(1) xargs -P $(JOBS) -I{} bash -o pipefail -c \
	'$(BLENDER) -b --factory-startup --python-exit-code 1 --python tests/run.py -- {} 2>&1 | grep -E "^(PASS|FAIL|ok|> )"'

unit:
	@$(if $(HAS_PYTEST),$(run_unit),$(NO_PYTEST); exit 1)

test:
	@$(if $(HAS_PYTEST),$(run_unit),$(NO_PYTEST); echo "unit tests skipped, the Blender tests run")
	@$(call blender_tests,)

test-slow:
	@$(if $(HAS_PYTEST),$(run_unit),$(NO_PYTEST); echo "unit tests skipped, the Blender tests run")
	@$(call blender_tests,SM_SLOW=1)

lint:
	@command -v $(RUFF) >/dev/null || { echo "no ruff: pip install ruff, or pass RUFF=/path/to/ruff"; exit 1; }
	$(RUFF) check . && $(RUFF) format --check .

build:
	mkdir -p dist && $(BLENDER) --command extension build --source-dir . --output-dir dist

release:
	BLENDER=$(BLENDER) MODEL_DIR=$(MODEL_DIR) RUNTIME_DIR=$(RUNTIME) ./release.sh $(LIBS)

.PHONY: unit test test-slow lint build release
