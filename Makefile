SHELL=bash

python:
	python setup.py build_ext --inplace

clean:
	python setup.py clean
	rm -rf C.2 vlasov/cython/*.{c,cpp,so,html}
	rm -rf vlasov/__pycache__ vlasov/cython/__pycache__
