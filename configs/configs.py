import os
import json

from ggpm.vocab import common_atom_vocab


class Configs(object):
    """Class to perform configs-parsing"""
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
        assert path.endswith('.json') is True

        with open(path, 'w') as file:
            json.dump(self.args, file)

    def from_json(self, json_str):
        configs = json.load(json_str)
        self.__dict__.update(configs)
        self.args = {k: v for k,v in self.__dict__.items()} # save args

        # set atom_vocab
        if self.atom_vocab_ is None:
            self.atom_vocab = common_atom_vocab

        # create saved_dir
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
