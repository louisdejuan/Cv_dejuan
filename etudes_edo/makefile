euler: main.o euler_imp.o euler_exp.o  bdf_ordre2.o tir_euler_imp.o newton.o f1.o
	gcc -Wall -o euler main.o euler_imp.o euler_exp.o  bdf_ordre2.o tir_euler_imp.o newton.o f1.o -lm

main.o: main.c euler_imp.h euler_exp.h  bdf_ordre2.h tir_euler_imp.h newton.h f1.h
	gcc -Wall -c main.c

euler_imp.o: euler_imp.c euler_imp.h
	gcc -Wall -c euler_imp.c

euler_exp.o: euler_exp.c euler_exp.h
	gcc -Wall -c euler_exp.c

bdf_ordre2.o: bdf_ordre2.c bdf_ordre2.h
	gcc -Wall -c bdf_ordre2.c	

tir_euler_imp.o: tir_euler_imp.c tir_euler_imp.h
	gcc -Wall -c tir_euler_imp.c

newton.o: newton.c newton.h
	gcc -Wall -c newton.c

f1.o: f1.c f1.h
	gcc -Wall -c f1.c



clean:
	   rm -f *.o euler