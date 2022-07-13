from enum import Enum


class Category(str, Enum):
    TRAINING = "train"
    VALIDATION = "val"
    TEST = "test"
