CC = cc
CFLAGS = -Wall -Wextra -O2 -std=c99

all: main

main: lm.c
	$(CC) $(CFLAGS) lm.c -o main

clean:
	rm -f main *.o

deps:
	git submodule update --remote --recursive

.PHONY: all clean deps