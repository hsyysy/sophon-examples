CC=gcc
LIBSOPHON_DIR=/opt/sophon/libsophon-current
main:main.c
	${CC} -o main main.c -I${LIBSOPHON_DIR}/include -L${LIBSOPHON_DIR}/lib -lbmrt -lbmlib -lm

clean:
	rm -rf main results
