PYTHON_VERSION := $(strip $(shell cat .python-version 2>/dev/null))
PYTHON_PREFIX = $(shell pyenv prefix $(PYTHON_VERSION) 2>/dev/null)
PYTHON_BIN = $(PYTHON_PREFIX)/bin/python
PROJECT_ROOT := $(CURDIR)
PROJECT_NAME := $(notdir $(PROJECT_ROOT))

.PHONY: install
install: pyenv_setup uv_setup
	@SITE_PACKAGES_DIR=$$(uv run python -c 'import sysconfig; print(sysconfig.get_path("purelib"))'); \
	echo "$(PROJECT_ROOT)" > "$${SITE_PACKAGES_DIR}/$(PROJECT_NAME).pth"

.PHONY: pyenv_setup
pyenv_setup:
	@test -n "$(PYTHON_VERSION)" || (echo "[ERROR] Failed to read Python version from .python-version."; exit 1)
	@pyenv install -s $(PYTHON_VERSION) || (echo "[ERROR] pyenv is not installed. See https://github.com/pyenv/pyenv#installation"; exit 1)

.PHONY: uv_setup
uv_setup:
	@test -n "$(PYTHON_PREFIX)" || (echo "[ERROR] Failed to obtain the prefix for $(PYTHON_VERSION) from pyenv."; exit 1)
	@test -x "$(PYTHON_BIN)" || (echo "[ERROR] $(PYTHON_BIN) not found. Verify 'pyenv install $(PYTHON_VERSION)'."; exit 1)
	@command -v uv >/dev/null 2>&1 || (echo "[ERROR] uv is not installed. See https://docs.astral.sh/uv/getting-started/installation/"; exit 1)
	@uv venv --python "$(PYTHON_BIN)"
	@uv sync --all-groups

.PHONY: run
run:
	bash bin/demo.sh

.PHONY: prepare
prepare:
	uv run python data/dysts

.PHONY: run_nohup
run_nohup:
	bash bin/demo.sh -n

.PHONY: freeze
freeze:
	uv pip compile pyproject.toml -o requirements.txt
