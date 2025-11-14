run-docker-compose: 
	uv sync
	docker compose up --build


run-evals-retriever:
	uv sync
	PYTHONPATH=${PWD}/src:$$PYTHONPATH:${PWD} uv run --env-file .env python -m evals.eval_retriever

run-evals-coordinator-agent: 
	uv sync
	PYTHONPATH=${PWD}/src:$$PYTHONPATH:${PWD} uv run --env-file .env python -m evals.eval_coordinator_agent