BLIS_INC    = /usr/local/include/blis
BLIS_LIB    = /usr/local/lib/libblis.a
OTHER_LIBS  = -L/usr/lib -lm -lpthread

CC 		    = gcc
TARGET      = test
C_FILES     = $(wildcard *.c)
OBJS        = $(patsubst %.c,%.o,$(C_FILES))
CFLAGS      = -g -I$(BLIS_INC)
LDFLAGS 	= 

.PHONY: all clean
all: $(TARGET)
$(TARGET): $(OBJS)
	$(CC) $(OBJS) $(LDFLAGS) $(CFLAGS) $(BLIS_LIB) $(OTHER_LIBS) -o $(TARGET)
%.o: %.c %.h
	$(CC) $(CFLAGS) -c -o $@ $<
clean:
	rm -f $(TARGET) $(TARGET).exe $(OBJS)
