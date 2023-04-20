import sys
import traceback
import string
import random


class Suppressor(object):
    # modified from https://stackoverflow.com/a/40054132

    def __enter__(self):
        self.stdout = sys.stdout
        sys.stdout = self

    def __exit__(self, type, value, traceback):
        sys.stdout = self.stdout
        if type is not None:
            pass
            # do normal exception handling

    def write(self, x):
        pass

    def flush(self):
        pass


def id_generator(size=8, chars=string.ascii_uppercase + string.ascii_lowercase + string.digits):
    # modified from https://stackoverflow.com/a/2257449
    return ''.join(random.choice(chars) for _ in range(size))
