import json
import typing
from dataclasses import MISSING, fields

# Minimal config handling without argparse: a dataclass defines the options, defaults and types.
#   python train.py                         # all defaults (required fields must be given)
#   python train.py config.json             # values from a json file
#   python train.py config.json lr=1e-3 keep_checkpoints=false     # command line overrides the file
#   python train.py dataset_path=data.npz results_path=out         # no file, only overrides


def _cast(value, tp):
    if typing.get_origin(tp) is typing.Union:
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if value.lower() == "none":
            return None
        tp = args[0]
    if tp is bool:
        if value.lower() not in ("true", "false"):
            raise ValueError(f"expected true or false, got {value!r}")
        return value.lower() == "true"
    return tp(value)


def load_config(cls, argv):
    """Instantiate dataclass `cls` from an optional json file and `key=value` overrides in argv."""
    hints = typing.get_type_hints(cls)
    names = {f.name for f in fields(cls)}
    values = {}
    for arg in argv:
        if "=" not in arg:
            with open(arg) as f:
                values.update(json.load(f))
            continue
        key, value = arg.split("=", 1)
        if key not in names:
            raise ValueError(f"unknown option {key!r}, available: {sorted(names)}")
        values[key] = _cast(value, hints[key])
    unknown = set(values) - names
    if unknown:
        raise ValueError(f"unknown options {sorted(unknown)}, available: {sorted(names)}")
    missing = [f.name for f in fields(cls) if f.name not in values and f.default is MISSING and f.default_factory is MISSING]
    if missing:
        raise ValueError(f"missing required options {missing}")
    return cls(**values)
