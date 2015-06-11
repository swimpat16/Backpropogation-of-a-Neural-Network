all: lab4 gen_config1 gen_config2 run

lab4:	lab4.cc
	g++ -g -O2 -o lab4 lab4.cc

gen_config1: gen_config1.cc
	g++ -g -o gen_config1 gen_config1.cc

gen_config2: gen_config2.cc
	g++ -g -o gen_config2 gen_config2.cc

run: run.cc
	g++ -g -o run run.cc

clean:
	rm -f lab4 gen_config1 gen_config2 run
