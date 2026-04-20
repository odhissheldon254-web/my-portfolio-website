CC     = gcc
CFLAGS = -O2 -Wall -std=c11 -I. -lm
SRC    = kernel/airos_kernel.c scheduler/rl_scheduler.c simulation/benchmark.c
OUT    = airos_bench

all:
	$(CC) $(CFLAGS) $(SRC) -o $(OUT)
	@echo "Build complete. Run: ./$(OUT)"

run: all
	./$(OUT)

clean:
	rm -f $(OUT) benchmark_results.csv
