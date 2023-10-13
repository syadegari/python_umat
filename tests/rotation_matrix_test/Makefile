FC = gfortran
FFLAGS = -Wunused-variable -O2 -g

all: generate rotation_matrix_fortran rotation_matrix_python compare

generate:
	$(FC) $(FFLAGS) generate_data.f90 -o generate_data
	./generate_data

rotation_matrix_fortran:
	$(FC) $(FFLAGS) rotation_matrix_fortran.f90 -o rotation_matrix_fortran
	./rotation_matrix_fortran

rotation_matrix_python:
	python rotation_matrix_python.py

compare:
	python compare_results.py

clean:
	rm -f generate_data rotation_matrix_fortran *.dat
	rm -rf *.dSYM
