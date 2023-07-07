# Go parameters
GOCMD=go
GOBUILD=$(GOCMD) build
GOCLEAN=$(GOCMD) clean
GOTEST=$(GOCMD) test
GOGET=$(GOCMD) get
BINARY_NAME=fast_bind
BINARY_UNIX=$(BINARY_NAME)_unix

all: clean test
build:
	$(GOBUILD) -o $(BINARY_NAME) -v

test:
	go run cli/main.go predict -m ../service/chat-quality/tools/ml/fasttext/model.bin "i accidentally shrunk my shrinky dink lol </s>"
	echo "i accidentally shrunk my shrinky dink lol" | ./fastText/fasttext predict-prob ../service/chat-quality/tools/ml/fasttext/model.bin -
	go run cli/main.go sentence -m ../service/chat-quality/tools/ml/fasttext/model.bin "i accidentally shrunk my shrinky dink lol </s>"
	echo "i accidentally shrunk my shrinky dink lol" | ./fastText/fasttext print-sentence-vectors ../service/chat-quality/tools/ml/fasttext/model.bin

clean:
	$(GOCLEAN)
	rm -f $(BINARY_NAME)
	rm -f $(BINARY_UNIX)

run:
	$(GOBUILD) -o $(BINARY_NAME) -v ./...
	./$(BINARY_NAME)

copy-fastText:
	cp -r $(PWD)/fastText/src/* ./fastText-src/
