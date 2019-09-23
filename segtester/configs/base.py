from segtester import logger
from datetime import datetime


class OptionalMember:
    def __init__(self, parsable_object=None, default_ret=None):
        parse_op = getattr(parsable_object, "_parse_args", None)
        assert (parsable_object is None or callable(parse_op)), f"provided object: {parsable_object} is not parsable"
        self.parsable_object = parsable_object
        self.default_ret = default_ret

    def _parse_args(self, key: str, trace: list, **kwargs):
        # noinspection PyBroadException
        try:
            logger.debug(f"Parsing member {' -> '.join(trace)}: {key}...")
            if key not in kwargs:
                return self.default_ret, True
            val = kwargs[key]

            if self.parsable_object is not None:
                # noinspection PyProtectedMember
                return self.parsable_object()._parse_args(key, trace, **kwargs)
            return val, True
        except Exception as e:
            logger.error(f"Unknown error occurred while parsing trace:\n{' -> '.join(trace)}")
            logger.error(str(e))
            return None, False


class RequiredMember(OptionalMember):
    def __init__(self, parsable_object=None):
        super().__init__(parsable_object)

    def _parse_args(self, key: str, trace: list, **kwargs):
        # noinspection PyBroadException
        try:
            if key not in kwargs:
                logger.error(f"Could not find required parameter: {key} with trace:\n{' -> '.join(trace)}")
                return None, False
            return super()._parse_args(key, trace, **kwargs)
        except Exception as e:
            logger.error(f"Unknown error occurred while parsing trace:\n{' -> '.join(trace)}")
            logger.error(str(e))
            return None, False


class ConstantMember:
    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self

    def _parse_args(self, key: str, trace: list, **kwargs):
        return self.value, False


class IterableMember:
    def __init__(self, parsable_object=None):
        parse_op = getattr(parsable_object, "_parse_args", None)
        assert (parsable_object is None or callable(parse_op))
        self.parsable_object = parsable_object

    def __call__(self):
        return self

    def _parse_args(self, key: str, trace: list, **kwargs):
        # noinspection PyBroadException
        try:
            logger.debug(f"Parsing IT {' -> '.join(trace)}: {key}...")
            conf_list = kwargs.get(key, [])
            ret = []
            success = True
            try:
                for i, item in enumerate(conf_list):
                    key_it = str(i)
                    if self.parsable_object is not None:
                        # noinspection PyProtectedMember
                        item, succ_it = self.parsable_object()._parse_args(key_it, trace + [key_it], **{key_it: item})
                        success = success and succ_it
                    ret.append(item)
                return ret, success
            except TypeError:
                logger.error(f"Expected iterable object:\n{' -> '.join(trace)}")
                return None, False
        except Exception as e:
            logger.error(f"Unknown error occurred while parsing trace:\n{' -> '.join(trace)}")
            logger.error(str(e))
            return None, False


class MappableMember:
    def __init__(self, type_map: dict):
        for key, value in type_map.items():
            parse_op = getattr(value, "_parse_args", None)
            assert (callable(parse_op)), f"Object registered for key: {key} is not parsable"
        self.type_map = type_map

    def __call__(self):
        return self

    def _parse_args(self, key: str, trace: list, **kwargs):
        # noinspection PyBroadException
        try:
            val = kwargs[key]
            if not isinstance(val, dict):
                logger.error(f"Expected a dictionary for {key}, trace: {' -> '.join(trace)}")
                return None, False

            if val.get('id') not in self.type_map:
                logger.error(f"Expected id to be one of: {set(self.type_map.keys())}\n -> trace: {' -> '.join(trace)}")
                return None, False
            # noinspection PyProtectedMember
            return self.type_map[val.get('id')]()._parse_args(key, trace, **kwargs)
        except Exception as e:
            logger.error(f"Unknown error occurred while parsing trace:\n{' -> '.join(trace)}")
            logger.error(str(e))
            return None, False


class SafeDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'


class ConfigParser:
    GLOBAL_META = {
        "timestamp": datetime.now().strftime('%Y%m%d-%H%M%S'),
    }

    def __init__(self):
        self.meta = OptionalMember(default_ret={})

    def _parse_args(self, key, trace: list, **kwargs):
        # noinspection PyBroadException
        try:
            logger.debug(f"Parsing conf {' -> '.join(trace)}: {key}...")
            curr_config = kwargs[key]
            if not isinstance(curr_config, dict):
                logger.error(f"Expected a dictionary for {key}, trace: {' -> '.join(trace)}")
                return None, False

            kwarg_keys = set(curr_config.keys())
            parse_success = True
            for key, value in self.__dict__.items():
                parse_op = getattr(value, "_parse_args", None)
                if not callable(parse_op):
                    logger.warn(f"Parameter: {key} with trace:\n{' -> '.join(trace)} is not parsable... Skipping!")
                    continue
                kwarg_keys.discard(key)
                # noinspection PyProtectedMember
                res_val, success = value._parse_args(key, trace + [key], **curr_config)
                parse_success = parse_success and success
                setattr(self, key, res_val)

            if len(kwarg_keys) > 0:
                logger.warn(f"Skipping unknown keys: {kwarg_keys}\n -> in config file: {' -> '.join(trace)}")
            return self, parse_success
        except Exception as e:
            logger.error(f"Unknown error occurred while parsing trace:\n{' -> '.join(trace)}")
            logger.error(str(e))
            return None, False

    def parse_from_config(self, init_trace: str, **kwargs):
        trace = []
        if init_trace is not None:
            trace += [init_trace]
        _, success = self._parse_args(init_trace, trace, **{init_trace: kwargs})
        assert success, "Invalid configurations detected please look at the logs."
        return self

    def parse_from_yaml_file(self, filename):
        from yaml import load, dump
        try:
            from yaml import CLoader as Loader
        except ImportError:
            from yaml import Loader
        with open(filename) as fp:
            config = load(fp, Loader=Loader)
        logger.info(f"Loading config from {filename}")
        return self.parse_from_config(filename, **config)

    def __repr__(self):
        from yaml import dump
        try:
            from yaml import CDumper as Dumper
        except ImportError:
            from yaml import Dumper
        return dump(self.to_dict(), Dumper=Dumper)

    def to_dict(self):
        ret = {}
        for key, value in self.__dict__.items():
            parse_op = getattr(value, "to_dict", None)
            if callable(parse_op):
                value = value.to_dict()
            ret[key] = value
        return ret

    @staticmethod
    def safe_format_string(string_template: str, **kwargs):
        return string_template.format_map(SafeDict(**kwargs))

    def format_string_with_meta(self, string_template: str, **kwargs):
        return self.safe_format_string(string_template, **{
            **ConfigParser.GLOBAL_META,
            **self.meta,
            **kwargs
        })
