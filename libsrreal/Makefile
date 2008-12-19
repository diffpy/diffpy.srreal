# User variables - can be set as environment variables or as make arguments
#
# DEBUG			debugging flags (-gstabs+), turns off optimizations

# Objcryst headers
ifndef OBJCRYST_HEADERS
OBJCRYST_HEADERS=/home/chris/local/src/ObjCryst/ObjCryst
endif 

# CFLAGS
ifdef DEBUG
CFLAGS = $(DEBUG)
else
CFLAGS = -fPIC -O3 -ffast-math -funroll-loops -Wall
endif

# Shared libary for so files
ifndef LIB
LIB = /home/chris/local/lib/
endif

# Shared library for python files
ifndef PYLIB
PYLIB = .
endif

# Location of included header files
INCDIR = .

# end of setup

OBJ = bonditerator.o PointsInSphere.o
SOBJ = libsrreal.so

CC        = g++
LINK      = g++
LINKFLAGS = -L$(LIB) -Wl,-O1


# Where to search for dependencies
VPATH = $(LIB):$(INCDIR)

.PHONY : install
install: lib
	cp $(SOBJ) $(LIB)

.PHONY : lib
lib: $(SOBJ)

# Make the profile library
$(SOBJ):$(OBJ)
	$(LINK) -shared $(LINKFLAGS) -o $(SOBJ) $(OBJ)

# rules for compiling libraries
%.o:%.cpp
	$(CC) $(CFLAGS) -c $< -I$(INCDIR) -I$(OBJCRYST_HEADERS)


# tests

TESTS = testsrreal

.PHONY : test
test: $(TESTS)

testsrreal: testsrreal.o
	make install
	$(LINK) $(CFLAGS) $(LINKFLAGS) -o $@ $^ -L. -lsrreal -lcryst -lCrystVector -lQuirks -lRefinableObj -lcctbx -lnewmat

.PHONY : clean
clean: 
	-rm -f *.so *.o $(TESTS)

