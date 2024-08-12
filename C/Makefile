CC=gcc
CFLAGS=-I ./include

all: obj bin validation

obj:
	@mkdir -p build
	@echo -e "Building Shared Objects..."
	$(CC) $(CFLAGS) -c layers/linear/linear_layer.c -o build/linear_layer.o
	$(CC) $(CFLAGS) -c layers/linear/feedforward.c -o build/linear_feedforward.o
	$(CC) $(CFLAGS) -c layers/linear/backpropagate.c -o build/linear_backpropagate.o
	$(CC) $(CFLAGS) -c layers/sigmoid/sigmoid_layer.c -o build/sigmoid_layer.o
	$(CC) $(CFLAGS) -c layers/sigmoid/feedforward.c -o build/sigmoid_feedforward.o
	$(CC) $(CFLAGS) -c layers/sigmoid/backpropagate.c -o build/sigmoid_backpropagate.o
	$(CC) $(CFLAGS) -c layers/relu/relu_layer.c -o build/relu_layer.o
	$(CC) $(CFLAGS) -c layers/relu/feedforward.c -o build/relu_feedforward.o
	$(CC) $(CFLAGS) -c layers/relu/backpropagate.c -o build/relu_backpropagate.o
	$(CC) $(CFLAGS) -c layers/leakyrelu/leakyrelu_layer.c -o build/leakyrelu_layer.o
	$(CC) $(CFLAGS) -c layers/leakyrelu/feedforward.c -o build/leakyrelu_feedforward.o
	$(CC) $(CFLAGS) -c layers/leakyrelu/backpropagate.c -o build/leakyrelu_backpropagate.o
	$(CC) $(CFLAGS) -c lib/common.c -o build/common.o
	$(CC) $(CFLAGS) -c lib/mnist.c -o build/mnist.o

bin:
	@mkdir -p build
	@echo -e "Building Examples..."
	$(CC) $(CFLAGS) -c examples/classifier.c -o build/classifier.o
	$(CC) -lm -o build/classifier build/mnist.o build/classifier.o build/common.o build/linear_layer.o build/linear_feedforward.o build/linear_backpropagate.o build/leakyrelu_layer.o build/leakyrelu_feedforward.o build/leakyrelu_backpropagate.o

validation:
	@mkdir -p build
	@echo -e "Building Tests..."
	$(CC) $(CFLAGS) -c tests/allocate-deallocate.c -o build/allocate-deallocate.o
	$(CC) $(CFLAGS) -c tests/mnist-load.c -o build/mnist-load.o
	$(CC) $(CFLAGS) -c tests/forward-backward.c -o build/forward-backward.o
	$(CC) $(CFLAGS) -c tests/matmult.c -o build/matmult.o
	$(CC) -lm -o build/allocate-deallocate-test build/allocate-deallocate.o build/common.o build/linear_layer.o build/linear_feedforward.o build/linear_backpropagate.o build/sigmoid_layer.o build/sigmoid_feedforward.o build/sigmoid_backpropagate.o
	$(CC) -lm -o build/mnist-load build/mnist-load.o build/common.o build/linear_layer.o build/linear_feedforward.o build/linear_backpropagate.o build/sigmoid_layer.o build/sigmoid_feedforward.o build/sigmoid_backpropagate.o build/mnist.o
	$(CC) -lm -o build/forward-backward build/forward-backward.o build/common.o build/linear_layer.o build/linear_feedforward.o build/linear_backpropagate.o build/relu_layer.o build/relu_feedforward.o build/relu_backpropagate.o build/mnist.o
	$(CC) -lm -o build/matmult build/matmult.o build/common.o build/linear_layer.o build/linear_feedforward.o build/linear_backpropagate.o build/relu_layer.o build/relu_feedforward.o build/relu_backpropagate.o build/mnist.o

clean:
	@echo -e "Cleanup..."
	@rm -Rf build linearckpt*.bin
