.PHONY: build test bench run clean

BIN := embedder
ADDR ?= :8089

build:
	go build -o $(BIN) ./cmd/text-embedder

test:
	go test ./... -v

bench:
	go test ./internal/embed/... -bench=. -benchmem -count=3

run: build
	./$(BIN) --addr $(ADDR)

clean:
	rm -f $(BIN)

# Cross-compile targets
build-linux:
	GOOS=linux GOARCH=amd64 go build -o $(BIN)-linux-amd64 ./cmd/text-embedder

build-mac:
	GOOS=darwin GOARCH=arm64 go build -o $(BIN)-darwin-arm64 ./cmd/text-embedder

build-windows:
	GOOS=windows GOARCH=amd64 go build -o $(BIN)-windows-amd64.exe ./cmd/text-embedder
