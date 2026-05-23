.PHONY: build test bench run clean compress deploy deploy-all

BIN := embedder
ADDR ?= :8089

# SocratiCode target directory (override if your layout differs)
SOCRA_DIR ?= ../SocratiCode

# Platform-specific names — deploy-all produces all three in one shot
LINUX_GZ   := text-embedder-linux.gz
MAC_GZ     := text-embedder-darwin.gz
WINDOWS_GZ := text-embedder-win.gz

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
	rm -f $(LINUX_GZ) $(MAC_GZ) $(WINDOWS_GZ)
	rm -f $(SOCRA_DIR)/$(LINUX_GZ) $(SOCRA_DIR)/$(MAC_GZ) $(SOCRA_DIR)/$(WINDOWS_GZ)

# ── Cross-compile + gzip ────────────────────────────────────────────────

.PHONY: linux-gz mac-gz windows-gz

linux-gz:
	GOOS=linux   GOARCH=amd64 go build -o $(BIN)-linux   ./cmd/text-embedder
	gzip -f --best $(BIN)-linux
	mv $(BIN)-linux.gz $(LINUX_GZ)

mac-gz:
	GOOS=darwin  GOARCH=arm64 go build -o $(BIN)-darwin  ./cmd/text-embedder
	gzip -f --best $(BIN)-darwin
	mv $(BIN)-darwin.gz $(MAC_GZ)

windows-gz:
	GOOS=windows GOARCH=amd64 go build -o $(BIN)-win.exe ./cmd/text-embedder
	gzip -f --best $(BIN)-win.exe
	mv $(BIN)-win.exe.gz $(WINDOWS_GZ)

# ── Deploy ──────────────────────────────────────────────────────────────

# Deploy a single native build (convenience for local dev)
deploy: linux-gz
	cp $(LINUX_GZ) $(SOCRA_DIR)/$(LINUX_GZ)
	@echo "Deployed $(LINUX_GZ) to $(SOCRA_DIR)/$(LINUX_GZ)"

# Deploy all three platform binaries at once
deploy-all: linux-gz mac-gz windows-gz
	cp $(LINUX_GZ)   $(SOCRA_DIR)/$(LINUX_GZ)
	cp $(MAC_GZ)     $(SOCRA_DIR)/$(MAC_GZ)
	cp $(WINDOWS_GZ) $(SOCRA_DIR)/$(WINDOWS_GZ)
	@echo "---"
	@echo "Deployed all 3 platform binaries to $(SOCRA_DIR)/:"
	@printf "  %-24s  %s\n" "$(LINUX_GZ)"   "$$(gzip -l $(SOCRA_DIR)/$(LINUX_GZ)   | awk 'NR==2{print $$2" bytes uncompressed"}')"
	@printf "  %-24s  %s\n" "$(MAC_GZ)"     "$$(gzip -l $(SOCRA_DIR)/$(MAC_GZ)     | awk 'NR==2{print $$2" bytes uncompressed"}')"
	@printf "  %-24s  %s\n" "$(WINDOWS_GZ)" "$$(gzip -l $(SOCRA_DIR)/$(WINDOWS_GZ) | awk 'NR==2{print $$2" bytes uncompressed"}')"
	@echo "(total on disk: $$(du -sh $(SOCRA_DIR)/text-embedder-*.gz | awk '{print $$1}'))"
