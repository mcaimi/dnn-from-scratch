CC=gcc
CFLAGS=-I ./include

all: obj bin

obj:
	@mkdir -p build
	$(CC) $(CFLAGS) -c layers/linear/linear_layer.c -o build/linear_layer.o
	$(CC) $(CFLAGS) -c layers/linear/feedforward.c -o build/linear_feedforward.o
	$(CC) $(CFLAGS) -c layers/linear/backpropagate.c -o build/linear_backpropagate.o
	$(CC) $(CFLAGS) -c layers/sigmoid/sigmoid_layer.c -o build/sigmoid_layer.o
	$(CC) $(CFLAGS) -c layers/sigmoid/feedforward.c -o build/sigmoid_feedforward.o
	$(CC) $(CFLAGS) -c layers/sigmoid/backpropagate.c -o build/sigmoid_backpropagate.o
	$(CC) $(CFLAGS) -c lib/common.c -o build/common.o
	$(CC) $(CFLAGS) -c tests/test.c -o build/test.o

bin:
	@mkdir -p build
	$(CC) -lm -o build/test build/test.o build/common.o build/linear_layer.o build/linear_feedforward.o build/linear_backpropagate.o build/sigmoid_layer.o build/sigmoid_feedforward.o build/sigmoid_backpropagate.o

clean:
	@rm -Rf build linearckpt*.bin
