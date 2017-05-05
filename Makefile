SHELL=bash

python:
	python setup.py build_ext --inplace

clean:
	python setup.py clean
	rm -rf C.2 vlasov/cython/*.{c,so,html}
	rm -rf vlasov/__pycache__ skeletor/cython/__pycache__
