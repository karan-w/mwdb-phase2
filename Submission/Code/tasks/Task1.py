import argparse

class Task1:
    def __init__(self):
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--x', type=str, required=True)
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--dimensionality_reduction_technique', type=str, required=True)

    args = parser.parse_args()