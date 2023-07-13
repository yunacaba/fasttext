# Go parameters
GOCMD=go
GOBUILD=$(GOCMD) build
GOCLEAN=$(GOCMD) clean
GOTEST=$(GOCMD) test
GOGET=$(GOCMD) get

test:
	@go test -covermode=atomic -coverprofile=coverage.txt -timeout 5m -json -v ./... | gotestfmt -showteststatus

clean:
	$(GOCLEAN)

copy-fastText:
	cp -r $(PWD)/fastText/src/* ./fastText-src/

.PHONY: tidy
tidy:
	@rm -f go.sum
	@go mod tidy

.PHONY: lint
lint:
	@golangci-lint run

.PHONY: fmt
fmt:
	@gofumpt -l -w .

gosec:
	@gosec ./...
