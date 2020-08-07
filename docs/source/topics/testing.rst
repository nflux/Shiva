# Unit tests

To run a Unit test session, run

```bash
pytest tests
```

## How to Contribute

For additional information, use https://docs.pytest.org/en/latest/

Write your own test cases and locate them on the [tests folder](../tests/).

The files, classes and methods that you want to be run on every test session, must have the word **test** on them. For example, if we have the below file tree, only 2 TestCases will be executed.

```
tests/
    .gitignore
    test_algorithms.py
    test_buffers.py
```

## Creating TestCases

Create a class inheriting the **unittest.TestCase** class.

Below sample is a good starting point.

```python
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
```