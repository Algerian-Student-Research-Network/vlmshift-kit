"""Task types: everything that depends on the shape of the question."""
from .yes_no import YesNoTask

TASKS = {"yes_no": YesNoTask}


def get_task(name: str):
    return TASKS[name]()
