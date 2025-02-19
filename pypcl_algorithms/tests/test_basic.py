import pypcl_algorithms as m


def test_main():
    assert m.__version__ == "0.0.1"
    assert m.add(1, 2) == 3
    assert m.subtract(1, 2) == -1
    print("All tests passed!")

if __name__ == "__main__":
    test_main()
