MODULEDIR = ./src/fewie
SCRIPT_DIR = ./scripts
TESTDIR = ./tests
ADDITIONAL_FILES = ./evaluate.py

.PHONY: quality style test

quality:
	black --check --line-length 100 --target-version py37 $(TESTDIR) $(MODULEDIR) $(ADDITIONAL_FILES)
	isort --check-only $(TESTDIR) $(MODULEDIR) $(ADDITIONAL_FILES)
	mypy $(MODULEDIR) $(ADDITIONAL_FILES) --ignore-missing-imports
	flake8 $(TESTDIR) $(MODULEDIR) $(ADDITIONAL_FILES)

# Format source code automatically

style:
	black --line-length 100 --target-version py37 $(TESTDIR) $(MODULEDIR) $(ADDITIONAL_FILES)
	isort $(TESTDIR) $(MODULEDIR) $(ADDITIONAL_FILES)

# Run tests for the library

test:
	python -m pytest -n 1 --dist=loadfile -s -v $(TESTDIR)
