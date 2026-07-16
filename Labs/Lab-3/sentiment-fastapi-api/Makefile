IMAGE_NAME := sentiment-fastapi-api
CONTAINER_NAME := sentiment-fastapi-api-container
HOST_PORT := 8000

.PHONY: build run clean

build:
	docker build -t $(IMAGE_NAME) .

run:
	docker run --rm --name $(CONTAINER_NAME) -p $(HOST_PORT):8000 $(IMAGE_NAME)

clean:
	-docker rm -f $(CONTAINER_NAME) 2>/dev/null
	-docker rmi -f $(IMAGE_NAME) 2>/dev/null
