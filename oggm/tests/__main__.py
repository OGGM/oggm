import os
import sys

HERE = os.path.dirname(__file__)


def main():
    import pytest
    return pytest.main([HERE] + sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
