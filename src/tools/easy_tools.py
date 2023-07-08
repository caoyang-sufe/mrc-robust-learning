# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn
import re
import time
import logging
from functools import wraps


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func_return = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"Function `{func.__name__}` runtime is {end_time - start_time} seconds.")
        return func_return

    return wrapper


def load_args(Config):
    config = Config()
    parser = config.parser
    try:
        return parser.parse_args()
    except Exception as exception:
        logging.warning(f"{exception}: are you in Jupyter Notebook?")
        return parser.parse_known_args()[0]


def save_args(args, save_path=None):
    class _MyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, type) or isinstance(obj, types.FunctionType):
                return str(obj)
            return json.JSONEncoder.default(self, obj)

    with open(save_path, 'w', encoding="utf8") as f:
        f.write(json.dumps(vars(args), cls=_MyEncoder))


def update_args(args, **kwargs):
    for key, value in kwargs.items():
        if key not in args:
            logging.warning(f"Key {key} not in args but you want to change its value to {value}!")
        args.__setattr__(key, value)


def initialize_logger(filename, mode='w'):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s | %(filename)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(filename, mode=mode, encoding="utf8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger


def terminate_logger(logger):
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
