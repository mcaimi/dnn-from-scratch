CC=gcc
CFLAGS=-I ./include

all: obj bin

obj:
	@mkdir -p build
	$(CC) $(CFLAGS) -c layers/linear/linear_layer.c -o build/layer.o
	$(CC) $(CFLAGS) -c layers/linear/feedforward.c -o build/feedforward.o
	$(CC) $(CFLAGS) -c layers/linear/backpropagate.c -o build/backpropagate.o
	$(CC) $(CFLAGS) -c lib/common.c -o build/common.o
	$(CC) $(CFLAGS) -c tests/test.c -o build/test.o

bin:
	@mkdir -p build
	$(CC) -o build/test build/test.o build/common.o build/layer.o build/feedforward.o build/backpropagate.o

clean:
	@rm -Rf build linearckpt*.bin
