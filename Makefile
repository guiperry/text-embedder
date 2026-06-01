.PHONY: build build-all test bench vet run clean compress deploy deploy-all

BIN_DIR := bin
BIN     := embedder
BIN_PATH := $(BIN_DIR)/$(BIN)
ADDR ?= :8089

# SocratiCode target directory (override if your layout differs)
SOCRA_DIR ?= ../SocratiCode

# Platform-specific names — deploy-all produces all three in one shot
LINUX_GZ   := text-embedder-linux.gz
MAC_GZ     := text-embedder-darwin.gz
WINDOWS_GZ := text-embedder-win.gz

build:
	mkdir -p $(BIN_DIR)
	go build -o $(BIN_PATH) ./cmd/text-embedder

test:
	go test ./... -v

bench:
	go test ./pkg/embed/... -bench=. -benchmem -count=3
	go test ./cmd/text-embedder/... -bench=. -benchmem -count=1

vet:
	go vet ./...

build-all:
	mkdir -p $(BIN_DIR)
	GOOS=linux   GOARCH=amd64 go build -o $(BIN_DIR)/$(BIN)-linux   ./cmd/text-embedder
	GOOS=darwin  GOARCH=arm64 go build -o $(BIN_DIR)/$(BIN)-darwin  ./cmd/text-embedder
	GOOS=windows GOARCH=amd64 go build -o $(BIN_DIR)/$(BIN)-win.exe ./cmd/text-embedder

run: build
	./$(BIN_PATH) --addr $(ADDR)

clean:
	rm -rf $(BIN_DIR)
	rm -f $(SOCRA_DIR)/$(LINUX_GZ) $(SOCRA_DIR)/$(MAC_GZ) $(SOCRA_DIR)/$(WINDOWS_GZ)

# ── Cross-compile + gzip ────────────────────────────────────────────────

.PHONY: linux-gz mac-gz windows-gz

linux-gz:
	mkdir -p $(BIN_DIR)
	GOOS=linux   GOARCH=amd64 go build -o $(BIN_DIR)/$(BIN)-linux   ./cmd/text-embedder
	gzip -f --best $(BIN_DIR)/$(BIN)-linux
	mv $(BIN_DIR)/$(BIN)-linux.gz $(BIN_DIR)/$(LINUX_GZ)

mac-gz:
	mkdir -p $(BIN_DIR)
	GOOS=darwin  GOARCH=arm64 go build -o $(BIN_DIR)/$(BIN)-darwin  ./cmd/text-embedder
	gzip -f --best $(BIN_DIR)/$(BIN)-darwin
	mv $(BIN_DIR)/$(BIN)-darwin.gz $(BIN_DIR)/$(MAC_GZ)

windows-gz:
	mkdir -p $(BIN_DIR)
	GOOS=windows GOARCH=amd64 go build -o $(BIN_DIR)/$(BIN)-win.exe ./cmd/text-embedder
	gzip -f --best $(BIN_DIR)/$(BIN)-win.exe
	mv $(BIN_DIR)/$(BIN)-win.exe.gz $(BIN_DIR)/$(WINDOWS_GZ)

# ── Deploy ──────────────────────────────────────────────────────────────

# Deploy a single native build (convenience for local dev)
deploy: linux-gz
	cp $(BIN_DIR)/$(LINUX_GZ) $(SOCRA_DIR)/$(LINUX_GZ)
	@echo "Deployed $(LINUX_GZ) to $(SOCRA_DIR)/$(LINUX_GZ)"

# Deploy all three platform binaries at once
deploy-all: linux-gz mac-gz windows-gz
	cp $(BIN_DIR)/$(LINUX_GZ)   $(SOCRA_DIR)/$(LINUX_GZ)
	cp $(BIN_DIR)/$(MAC_GZ)     $(SOCRA_DIR)/$(MAC_GZ)
	cp $(BIN_DIR)/$(WINDOWS_GZ) $(SOCRA_DIR)/$(WINDOWS_GZ)
	@echo "---"
	@echo "Deployed all 3 platform binaries to $(SOCRA_DIR)/:"
	@printf "  %-24s  %s\n" "$(LINUX_GZ)"   "$$(gzip -l $(SOCRA_DIR)/$(LINUX_GZ)   | awk 'NR==2{print $$2" bytes uncompressed"}')"
	@printf "  %-24s  %s\n" "$(MAC_GZ)"     "$$(gzip -l $(SOCRA_DIR)/$(MAC_GZ)     | awk 'NR==2{print $$2" bytes uncompressed"}')"
	@printf "  %-24s  %s\n" "$(WINDOWS_GZ)" "$$(gzip -l $(SOCRA_DIR)/$(WINDOWS_GZ) | awk 'NR==2{print $$2" bytes uncompressed"}')"
	@echo "(total on disk: $$(du -sh $(SOCRA_DIR)/text-embedder-*.gz | awk '{print $$1}'))"
