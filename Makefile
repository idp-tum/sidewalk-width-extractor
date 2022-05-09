.PHONY: default
default: dev

.PHONY: install
install:
	python setup.py install

.PHONY: dev
dev:
	python setup.py develop

.PHONY: clean
clean:
	python setup.py develop --uninstall
	rm -rf build