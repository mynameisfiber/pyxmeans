CC := gcc
CFLAGS := -O3 -std=c99 -fopenmp -DVERBOSE -Wall
LDFLAGS := -lgomp -lc

OBJECTS := $(addsuffix .o, $(basename $(wildcard *.c)))

minibatch: $(OBJECTS)
	$(CC) $^ -o minibatch $(LDFLAGS)

%.o: %.c %.h 
	$(CC) $< -c $(CFLAGS)

clean:
	rm -rf *.o minibatch minibatch.dSYM
