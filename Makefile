develop: ## Build the Rust extension into the local venv
	@uv run maturin develop --uv

test: develop ## Build and run the test suite
	@uv run pytest

lint: ## Run the linter and type checker
	@uv run ruff format
	@uv run ruff check
	@uv run ty check
