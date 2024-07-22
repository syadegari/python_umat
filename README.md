# About

This is the repository for the work on physics-informed prediction of the crystal plasticity model, as explained in [Tjahjanto D.D. 2008](https://repository.tudelft.nl/person/Person_143ffcdb-f1d0-424f-8d39-70b9f581ee27).

# Download

To download the repository:

```bash
git clone -b replay_buffer https://github.com/syadegari/python_umat.git
```

Note: The repository uses a submodule that contains the physics engine, which implements the evolution of the crystal plasticity model. The mentioned repository cannot be publicly shared before the publication of the current work. If you need to access the physics engine, you need to contact the repository maintainer.

# Installation

You can install the project locally (in editable mode) using `pip` as follows.

```bash
pip install -e .
```

# Documentation

You will find partial documentation and a background of the physics model under the docs folder. You need to compile that using LaTeX in order to render it into a PDF. Please note that this is currently a developer documentation that is used for implementing different aspects of the physical model and various algorithms in the project.

# Testing

To perform the testing, run:

```bash
python -m unittest tests.test_umat -v
```

All the above tests should pass without the need to access the physics engine. Several specific tests are implemented that ensure the compatibility and correctness of functions both in the engine and their counterparts in the Python functions, each in their own subdirectories in the tests folder. For those, one needs to have access to the physics engine.
