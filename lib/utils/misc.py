import ctypes
import functools
import math
import os
import re
import sys
from collections import namedtuple

import yaml
from termcolor import colored

bar_prefix = {
    "train": colored("train", "white", attrs=["bold"]),
    "val": colored("val", "yellow", attrs=["bold"]),
    "test": colored("test", "magenta", attrs=["bold"]),
}

RandomState = namedtuple(
    "RandomState",
    [
        "torch_rng_state",
        "torch_cuda_rng_state",
        "torch_cuda_rng_state_all",
        "numpy_rng_state",
        "random_rng_state",
    ],
)
RandomState.__new__.__default__ = (None,) * len(RandomState._fields)


def enable_lower_param(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kw_uppers = {}
        for k, v in kwargs.items():
            kw_uppers[k.upper()] = v
        return func(*args, **kw_uppers)

    return wrapper


def singleton(cls):
    _instance = {}

    @functools.wraps(cls)
    def inner(*args, **kwargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]

    return inner


class RedirectStream(object):

    @staticmethod
    def _flush_c_stream(stream):
        streamname = stream.name[1:-1]
        libc = ctypes.CDLL(None)
        libc.fflush(ctypes.c_void_p.in_dll(libc, streamname))

    def __init__(self, stream=sys.stdout, file=os.devnull):
        self.stream = stream
        self.file = file

    def __enter__(self):
        self.stream.flush()  # ensures python stream unaffected
        self.fd = open(self.file, "w+")
        self.dup_stream = os.dup(self.stream.fileno())
        os.dup2(self.fd.fileno(), self.stream.fileno())  # replaces stream

    def __exit__(self, type, value, traceback):
        RedirectStream._flush_c_stream(self.stream)  # ensures C stream buffer empty
        os.dup2(self.dup_stream, self.stream.fileno())  # restores stream
        os.close(self.dup_stream)
        self.fd.close()


class ImmutableClass(type):

    def __call__(cls, *args, **kwargs):
        raise AttributeError("Cannot instantiate this class")

    def __setattr__(cls, name, value):
        raise AttributeError("Cannot modify immutable class")

    def __delattr__(cls, name):
        raise AttributeError("Cannot delete immutable class")


class CONST(metaclass=ImmutableClass):
    PI = math.pi
    INT_MAX = 2**32 - 1
    NUM_JOINTS = 21

    MANO_KPID_2_VERTICES = {
        4: [744],  #ThumbT
        8: [320],  #IndexT
        12: [443],  #MiddleT
        16: [555],  #RingT
        20: [672]  #PinkT
    }


def update_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        return config


def format_cfg(cfg={}, indent=0):
    cfg_str = ""

    for k, v in cfg.items():
        if not isinstance(v, dict):
            cfg_str += f"\n{'  '*indent} - {colored(k, 'magenta')}: {v}"
        else:
            cfg_str += f"\n{'  '*indent} - {colored(k, 'magenta')}: {format_cfg(v, indent+1)}"
    return cfg_str


def format_args_cfg(args, cfg={}):
    args_list = [f" - {colored(name, 'green')}: {getattr(args, name)}" for name in vars(args)]
    arg_str = "\n".join(args_list)
    cfg_str = format_cfg(cfg)
    return arg_str + cfg_str


def camel_to_snake(camel_input):
    words = re.findall(r'[A-Z]?[a-z]+|[A-Z]{1,}(?=[A-Z][a-z]|\d|\W|$)|\d+', camel_input)
    return '_'.join(map(str.lower, words))
