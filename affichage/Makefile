# include Make_linux.inc
include Make_msys2.inc
#include Make_osx.inc

CXXFLAGS = -std=c++17 `sdl2-config --cflags` -fconcepts
ifdef DEBUG
CXXFLAGS += -g -O0 -Wall -fbounds-check -pedantic -D_GLIBCXX_DEBUG -DDEBUG
else
CXXFLAGS += -O3 -march=native -Wall
endif

LIBS = $(LIB) `sdl2-config --cflags` -lSDL2_ttf -lSDL2_image  `sdl2-config --libs`
ALL = affichage.exe

default: $(ALL)

all: $(ALL)

clean:
	@rm -fr SDL2/*.o *.exe SDL2/*~

OBJS = SDL2/sdl2.o SDL2/geometry.o SDL2/window.o SDL2/font.o SDL2/event.o SDL2/texte.o SDL2/image.o SDL2/formated_text.o

SDL2/sdl2.o:	SDL2/sdl2.hpp SDL2/sdl2.cpp
	$(CXX) $(CXXFLAGS) -c SDL2/sdl2.cpp -o SDL2/sdl2.o

SDL2/geometry.o:	SDL2/geometry.hpp SDL2/window.hpp SDL2/geometry.cpp
	$(CXX) $(CXXFLAGS) -c SDL2/geometry.cpp -o SDL2/geometry.o

SDL2/window.o:	SDL2/window.hpp SDL2/window.cpp
	$(CXX) $(CXXFLAGS) -c SDL2/window.cpp -o SDL2/window.o

SDL2/font.o:	SDL2/window.hpp SDL2/font.hpp SDL2/font.cpp
	$(CXX) $(CXXFLAGS) -c SDL2/font.cpp -o SDL2/font.o

SDL2/event.o : SDL2/window.hpp SDL2/event.hpp SDL2/event.cpp
	$(CXX) $(CXXFLAGS) -c SDL2/event.cpp -o SDL2/event.o

SDL2/texte.o : SDL2/window.hpp SDL2/font.hpp SDL2/texte.hpp SDL2/texte.cpp
	$(CXX) $(CXXFLAGS) -c SDL2/texte.cpp -o SDL2/texte.o

SDL2/formated_texte.o : SDL2/window.hpp SDL2/font.hpp SDL2/formated_text.hpp SDL2/formated_text.cpp
	$(CXX) $(CXXFLAGS) -c SDL2/formated_text.cpp -o SDL2/formated_text.o

SDL2/image.o : SDL2/window.hpp SDL2/image.hpp SDL2/image.cpp
	$(CXX) $(CXXFLAGS) -c SDL2/image.cpp -o SDL2/image.o

affichage.exe : $(OBJS) affichage.cpp
	make -f ../structures_Centrale/Makefile ../structures_Centrale/centrale.o ../structures_Centrale/rnd.o ../structures_Centrale/dispatching.o ../structures_Centrale/catastrophes.o
	$(CXX) $(CXXFLAGS) $(OBJS) ../structures_Centrale/centrale.o ../structures_Centrale/rnd.o ../structures_Centrale/dispatching.o ../structures_Centrale/catastrophes.o affichage.cpp -o affichage.exe $(LIBS)


show:
	@echo "LIB   : $(LIB)"
	@echo "LIBS  : $(LIBS)"
