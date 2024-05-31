.PHONY: biuld
biuld:
	docker compose build

.PHONY: start
start:
	docker compose up -d
