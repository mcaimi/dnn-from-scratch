OS := $(shell uname)

ifeq ($(OS), Darwin)
	CC := clang
	ECHO := gecho
else
	CC := gcc
	ECHO := echo
endif

SRCDIR := $(shell find src -type d)
INCLUDEDIR := include
BINDIR := bin
SRCFILES := $(foreach D, $(SRCDIR), $(wildcard $(D)/*.c))
OBJFILES := $(patsubst %.c,%.o,$(SRCFILES))
EXAMPLESOURCE := $(wildcard examples/*.c)
EXAMPLEOBJ := $(patsubst %.c,%.o,$(EXAMPLESOURCE))
EXAMPLEBIN := $(patsubst %.o,%,$(EXAMPLEOBJ))
TESTSOURCE := $(wildcard tests/*.c)
TESTOBJ := $(patsubst %.c,%.o,$(TESTSOURCE))
TESTBIN := $(patsubst %.o,%,$(TESTOBJ))
CFLAGS := -I $(INCLUDEDIR)
LDFLAGS := -lm

all: obj examples tests

obj: $(OBJFILES)

examples: $(EXAMPLEOBJ) $(EXAMPLEBIN)

tests: $(TESTOBJ) $(TESTBIN)

%.o: %.c
	@$(ECHO) -e "Building Shared Object [$@]..."
	@$(CC) $(CFLAGS) -c $< -o $@

%: %.o | $(BINDIR)
	@$(ECHO) -e "Building executable [$@]..."
	@$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $(BINDIR)/$(shell basename $@) $(OBJFILES)

$(BINDIR):
	@mkdir -p $@

clean:
	@$(ECHO) -e "Cleanup binary objects..."
	@rm -f $(OBJFILES)
	@rm -f $(EXAMPLEOBJ)
	@rm -f $(TESTOBJ)
	@$(ECHO) -e "Cleaning executables..."
	@rm -Rf $(BINDIR)
