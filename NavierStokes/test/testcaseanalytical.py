import unittest
import numpy as np
# import warnings
# warnings.simplefilter(action="error", category=DeprecationWarning)
#================================================================#
class TestCaseAnalytical(unittest.TestCase):
    failed = []
    def __init__(self, args):
        super().__init__()
        self.args = args
    def checkerrors(self, errors, eps=1e-10):
        # print(f"{next(iter(errors.values())).keys()} {errors.keys()}")
        failed = {}
        for meth,err in errors.items():
            assert isinstance(err, dict)
            for m, e in err.items():
                if not np.all(e < eps): failed[meth] = e
        if len(failed):
            TestCaseAnalytical.failed.append(self.args)
            self.fail(msg=f'Test case failed {self.args=}\n{failed=}')

#================================================================#
def run(testcase, argss=None):
    import os, json
    filename = f"{testcase.__name__}_Failed.txt"
    # print(f"$$$$$$$$$$$$$$$ {filename=}")
    if os.path.exists(filename) and os.path.getsize(filename) > 2:
        # check if log-file exists and contains more than empty list
        with open(filename, 'r') as f:
            argss = json.loads(f.read())
    suite = unittest.TestSuite()
    TestCaseAnalytical.failed = []
    for args in argss:
        suite.addTest(testcase(args))
    unittest.TextTestRunner().run(suite)
    with open(filename, 'w') as f:
        f.write(json.dumps(TestCaseAnalytical.failed))
    # print(f"{TestCaseAnalytical.failed=}")
