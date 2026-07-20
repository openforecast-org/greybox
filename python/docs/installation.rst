============
Installation
============

Requirements
------------

greybox requires:

* Python >= 3.8
* numpy >= 1.20.0
* scipy >= 1.7.0
* pandas >= 1.3.0
* nlopt >= 2.7.0

Install from PyPI
-----------------

The recommended way to install greybox is via pip::

    pip install greybox

Install from Source
-------------------

To install the latest development version from GitHub::

    git clone https://github.com/openforecast-org/greybox.git
    cd greybox/python
    pip install -e .

Install with Documentation
--------------------------

To install with documentation dependencies::

    pip install -e ".[docs]"

Then build the documentation::

    sphinx-build -b html docs docs/_build/html

Optional Dependencies
---------------------

For development and testing::

    pip install -e ".[dev]"

This installs:

* pytest - for running tests
* flake8 - for code linting
* mypy - for type checking
* black - for code formatting
