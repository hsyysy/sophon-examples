CC = gcc
LIBSOPHON_DIR = /opt/sophon/libsophon-current
CFLAGS = -O3
#CFLAGS += -g

# 检查是否支持 sse
SSE_SUPPORTED := $(shell lscpu | grep -q 'sse4_1' && echo "yes" || echo "no")

ifeq ($(SSE_SUPPORTED), yes)
    CFLAGS += -msse4.1
    $(info SSE4.1 is supported)
endif

main:main.c utils.h
	${CC} $(CFLAGS) -o $@ main.c -I${LIBSOPHON_DIR}/include -L${LIBSOPHON_DIR}/lib -lbmrt -lbmlib -lm

clean:
	rm -rf main results
