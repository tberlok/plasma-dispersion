python:
	python setup.py build_ext --inplace

figures/%.eps: examples/*.py
	cd examples; $(MAKE) ../$@

figures = L_mode_damping fast_wave oblique_firehose ion_cyclotron \
oblique_mirror R_mode_damping kinetic_alfven parallel_firehose \
bernstein ion_acoustic slow_wave
figures: $(addprefix figures/, $(addsuffix .eps, $(figures)))

clean:
	python setup.py clean
	rm -rf C.2 vlasov/cython/*.{c,cpp,so,html}
	rm -rf vlasov/__pycache__ vlasov/cython/__pycache__

cleanest:
	rm -rf figures/*.eps
