CC=gcc
CFLAGS=-I ./include

all: obj bin

obj:
	@mkdir -p build
	$(CC) $(CFLAGS) -c neuron/neuron_layer.c -o build/layer.o
	$(CC) $(CFLAGS) -c lib/common.c -o build/common.o
	$(CC) $(CFLAGS) -c operators/feedforward.c -o build/feedforward.o
	$(CC) $(CFLAGS) -c tests/test.c -o build/test.o

bin:
	@mkdir -p build
	$(CC) -o build/test build/test.o build/common.o build/layer.o build/feedforward.o

clean:
	@rm -Rf build neuronckpt*.bin
