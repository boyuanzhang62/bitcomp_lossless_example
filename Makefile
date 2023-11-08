bitcomp_example: bitcomp_example.cu
	nvcc -L/home/bozhan/repo/nvcomp/lib -I /home/bozhan/repo/nvcomp/include/ bitcomp_example.cu -lcuda -lnvcomp_bitcomp -o bitcomp_example

clean:
	rm -rf bitcomp_example