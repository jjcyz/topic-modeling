def test_imports():
    """Test if all required packages are properly installed."""
    try:
        import numpy
        print("numpy version:", numpy.__version__)
    except ImportError:
        print("numpy not found")

    try:
        import scipy
        print("scipy version:", scipy.__version__)
    except ImportError:
        print("scipy not found")

    try:
        import sklearn
        print("sklearn version:", sklearn.__version__)
    except ImportError:
        print("sklearn not found")

    try:
        import matplotlib
        print("matplotlib version:", matplotlib.__version__)
    except ImportError:
        print("matplotlib not found")

    try:
        import seaborn
        print("seaborn version:", seaborn.__version__)
    except ImportError:
        print("seaborn not found")

if __name__ == "__main__":
    print("Testing package imports...")
    test_imports()
