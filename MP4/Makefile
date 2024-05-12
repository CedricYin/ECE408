WB = ${WB_DIR}

template.o: template.cu
	nvcc -std=c++11 -rdc=true -I $(WB) -c template.cu -o template.o

solution: template.o
	nvcc -std=c++11 -o solution template.o $(WB)/lib/libwb.so

test:
	@for i in $$(seq 0 5);do echo "---------------";echo Dataset $$i;\
	./solution -e ./data/$$i/output.dat -i ./data/$$i/input.dat,./data/$$i/kernel.dat -o ./data/$$i/myoutput.dat -t vector; done

clean:
	-rm -f template.o
	-rm -f solution
	-rm data/**/myoutput*