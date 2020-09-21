.. _trackstream-test:

=================
Running the tests
=================

The tests are written assuming they will be run with `pytest <http://doc.pytest.org/>`_ using the Astropy `custom test runner <http://docs.astropy.org/en/stable/development/testguide.html>`_. To set up a Conda environment to run the full set of tests, install ``trackstream`` or see the setup.cfg file for dependencies. Once the dependencies are installed, you can run the tests two ways:

1. By importing ``trackstream``::

    import trackstream
    trackstream.test()

2. By cloning the ``trackstream`` repository and running::

    python setup.py test
