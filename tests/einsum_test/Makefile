FC = gfortran
FFLAGS = -Wunused-variable -O2 -g

all: generate rotate_fortran rotate_python compare

generate:
	$(FC) $(FFLAGS) generate_data.f90 -o generate_data
	./generate_data

rotate_fortran:
	$(FC) $(FFLAGS) rotate_fortran.f90 -o rotate_fortran
	./rotate_fortran

rotate_python:
	python rotate_python.py

compare:
	python compare_results.py

clean:
	rm -f generate_data rotate_fortran rotation.dat slipsys.dat output_*.dat
	rm -rf *.dSYM
