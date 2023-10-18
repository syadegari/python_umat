import unittest


class CustomTestResults(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super(CustomTestResults, self).__init__(*args, **kwargs)
        self.current_class = None

    def startTest(self, test):
        # If the current test class has changed, print it out
        test_class = test.__class__
        if self.current_class != test_class:
            self.stream.writeln(test_class.__name__)
            self.current_class = test_class
        super(CustomTestResults, self).startTest(test)

    def getDescription(self, test):
        return "    " + test._testMethodName


class TestRunner(unittest.TextTestRunner):
    resultclass = CustomTestResults


if __name__ == "__main__":
    # Discover all tests
    test_suite = unittest.defaultTestLoader.discover(start_dir="tests", pattern="test_*.py")

    # Run the tests using the custom test runner
    runner = TestRunner(verbosity=2)
    runner.run(test_suite)
