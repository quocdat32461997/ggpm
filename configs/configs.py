import os
import json

from ggpm.vocab import common_atom_vocab

class Configs(object):
    def __init__(self, path=None, args=None):
        if path is not None:
            assert path.endswith('.json') and os.path.exists(path) is True
            self.from_json(open(path, 'r'))
        elif args is not None:
            assert isinstance(args, dict) is True
            self.from_json(args)
        else:
            raise "Either path or args must be a valid value"

    def to_json(self, path):
        assert path.endswith('.json') and os.path.exists(path) is True
        with  open(path, 'w') as file:
            json.dumps(self.__dict__)

    def from_json(self, json_str):
        configs = json.load(json_str)
        self.__dict__.update(configs)

        if self.atom_vocab is None:
            self.atom_vocab = common_atom_vocab

