import pprint


def say(*args, **kwargs):
    print("[ARBOK]", *args, **kwargs)


def log(*args, **kwargs):
    print(*args, **kwargs)


def pretty(*args, **kwargs):
    pprint.PrettyPrinter(indent=4).pprint(*args, **kwargs)


def header(name):
    print(f"\n{name}\n{'-' * len(name)}")
