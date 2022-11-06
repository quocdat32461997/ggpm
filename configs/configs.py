import os
import json

class Configs(object):
    """Class to perform configs-parsing"""
    def __init__(self, path=None, args=None):
        self.args = args
        if path is not None:
            assert path.endswith('.json') and os.path.exists(path) is True
            with open(path, 'r') as file:
                self.from_json(json.load(file))
        elif args is not None:
            assert isinstance(args, dict) is True
            self.from_json(args)
        else:
            raise Exception("Either path or args must be a valid value")

    def to_json(self, path):
        assert isinstance(path, str) and path.endswith('.json')

        with open(path, 'w') as file:
            json.dump(self.args, file)

    def from_json(self, configs):
        self.__dict__.update(configs)
        self.args = {k: v for k,v in self.__dict__.items()} # save args

        # set atom_vocab
        if 'atom_vocab_' in configs and self.atom_vocab_ is None:
            from ggpm.vocab import common_atom_vocab
            self.atom_vocab = common_atom_vocab

        # create saved_dir
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
