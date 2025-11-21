# Default recipe
default:
    @just --list

# Train a model
train:
    uv run tinify train image -d dummy_dataset -e 50 --batch-size 1 --patch-size 128 128 --model mbt2018-mean

# Run all linting and type checking
lint:
    uv run ruff check --fix .
    uv run ruff format .

# Type check with ty (core module only)
ty:
    uv run ty check tinify/

# Type check with zuban (core module only)
zuban:
    uv run zuban check tinify/

# Run both type checkers
typecheck: ty zuban

# Run prek (pre-commit hooks)
prek:
    uv run prek run --all-files

# Install prek hooks
prek-install:
    uv run prek install

# Run tests
test:
    uv run pytest tests/ -v

# Run tests with coverage
test-cov:
    uv run pytest tests/ -v --cov=tinify --cov-report=term-missing

# Full check: lint, typecheck, and test
check: lint typecheck test

# Quick check: just lint and typecheck
quick-check: lint typecheck
