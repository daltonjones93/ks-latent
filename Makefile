.PHONY: env-check test test-fast smoke replicate bench

ENV := da_env
RUN := mamba run -n $(ENV)

env-check:
	$(RUN) python scripts/check_env.py

test-fast:
	$(RUN) python -m pytest tests/unit tests/integration -m "not slow" -v

test:
	$(RUN) python -m pytest tests -v

smoke:
	$(RUN) python -m pytest tests/integration -v

replicate:
	$(RUN) python -m pytest tests/replication -v -m slow
	$(RUN) python scripts/generate_replication_log.py

bench:
	$(RUN) python scripts/bench_device.py
