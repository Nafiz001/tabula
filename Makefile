.PHONY: install test slow lint demo clean

install:
	pip install -e ".[dev]"

test:
	pytest

slow:
	pytest -m slow -v

# Every headline claim in the README, reproduced.
demo:
	tabula cliff
	tabula train grid
	tabula train tictactoe
	tabula train cartpole
	tabula train alphazero

clean:
	rm -rf .pytest_cache __pycache__ */__pycache__ */*/__pycache__ *.egg-info
