==========
Unit Tests
==========

To run a Unit test session, run

.. code-block:: bash

   pytest tests


.. rubric:: How to Contribute

For additional information, use https://docs.pytest.org/en/latest/

Write your own test cases and locate them in the `tests folder <https://github.com/nflux/Control-Tasks/tree/docs/shiva/tests>`_.

The files, classes and methods that you want to be run on every test session, must have the word **test** on them. For example, if we have the below file tree, only 2 TestCases will be executed.

.. note::
   tests/
       .gitignore
       test_algorithms.py
       test_buffers.py

.. ruric:: Creating TestCases

Create a class inheriting the **unittest.TestCase** class.

Below sample is a good starting point.

.. code-block:: python

   from unittest import TestCase
   
   class test_algorithms(TestCase):
       def tearDown(self):
       '''
           This method is run after all the tests are finished
       '''
   
       def setUp(self):
       '''
           This method is run before all the tests execution
           For example, prepare some data structures shared among the tests in this TestCase
       '''
   
       def test_1(self):
           ...
           ...
       
       def test_2(self):
           ...
           ...
