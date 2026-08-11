API_IMAGE := assignment5-sentiment-api
MONITOR_IMAGE := assignment5-monitoring-dashboard
API_CONTAINER := sentiment-api
MONITOR_CONTAINER := sentiment-monitoring
NETWORK := sentiment-monitoring-network
VOLUME := sentiment-prediction-logs
API_PORT := 8000
MONITOR_PORT := 8501

.PHONY: build run clean clean-all status logs evaluate

build:
	docker build -t $(API_IMAGE) ./api
	docker build -t $(MONITOR_IMAGE) ./monitoring

run:
	@docker network inspect $(NETWORK) >/dev/null 2>&1 || docker network create $(NETWORK)
	@docker volume inspect $(VOLUME) >/dev/null 2>&1 || docker volume create $(VOLUME)
	-@docker rm -f $(API_CONTAINER) >/dev/null 2>&1
	-@docker rm -f $(MONITOR_CONTAINER) >/dev/null 2>&1
	docker run -d \
		--name $(API_CONTAINER) \
		--network $(NETWORK) \
		-p $(API_PORT):8000 \
		-v $(VOLUME):/logs \
		$(API_IMAGE)
	docker run -d \
		--name $(MONITOR_CONTAINER) \
		--network $(NETWORK) \
		-p $(MONITOR_PORT):8501 \
		-v $(VOLUME):/logs:ro \
		-e API_SERVICE_URL=http://$(API_CONTAINER):8000 \
		$(MONITOR_IMAGE)
	@echo "FastAPI:   http://localhost:$(API_PORT)"
	@echo "API docs:  http://localhost:$(API_PORT)/docs"
	@echo "Dashboard: http://localhost:$(MONITOR_PORT)"

clean:
	-@docker rm -f $(MONITOR_CONTAINER) >/dev/null 2>&1
	-@docker rm -f $(API_CONTAINER) >/dev/null 2>&1
	-@docker network rm $(NETWORK) >/dev/null 2>&1
	-@docker volume rm $(VOLUME) >/dev/null 2>&1
	@echo "Containers, shared network, and shared log volume removed."

clean-all: clean
	-@docker rmi -f $(MONITOR_IMAGE) >/dev/null 2>&1
	-@docker rmi -f $(API_IMAGE) >/dev/null 2>&1
	@echo "Docker images removed."

status:
	@docker ps --filter name=$(API_CONTAINER) --filter name=$(MONITOR_CONTAINER)

logs:
	docker logs $(API_CONTAINER)
	@echo ""
	docker logs $(MONITOR_CONTAINER)

evaluate:
	python3 evaluate.py
