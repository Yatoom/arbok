import pprint

import time
import uuid

activities = {}


def say(*args, **kwargs):
    print("[ARBOK]", *args, **kwargs)


def log(*args, **kwargs):
    print(*args, **kwargs)


def pretty(*args, **kwargs):
    pprint.PrettyPrinter(indent=4).pprint(*args, **kwargs)
    print()


def header(name):
    print(f"\n{name}\n{'-' * len(name)}")


def start(name):
    random_id = uuid.uuid4()
    activities[random_id] = time.time()

    say(f"START: {name}")
    return random_id


def done(name, activity_id):
    duration = time.time() - activities.pop(activity_id)
    say(f"FINISHED: {name} - {duration:2f} seconds.")


def fail(name, activity_id):
    duration = time.time() - activities.pop(activity_id)
    say(f"FAILED: {name} - {duration:2f} seconds.")
