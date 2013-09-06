CC := llvm-gcc-4.2
CFLAGS := -O3 -std=c99 -fopenmp -DVERBOSE -Wall -lc

OBJECTS := $(basename $(wildcard *.c))

all: $(OBJECTS)

%: %.c
	$(CC) $< $(CFLAGS) -o $@

clean:
	rm -rf $(OBJECTS) *.dSYM
