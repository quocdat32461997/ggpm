import argparse

def merge(args):
    # read the 1st vocab file
    print('Read the 1st vocab file: {}.'.format(args.vocabs[0]))
    with open(args.vocabs[0]) as file:
        vocabs = file.read().split('\n')
        # remove last item if empty
        if len(vocabs[-1]) == 0:
            vocabs = vocabs[::-1]

    # read remaining vocab files
    # and get a set of unique vocabs
    _vocabs = []
    for x in args.vocabs[1:]:
        print('Read other vocab file: {}.'.format(x))
        with open(x) as file:
            _vocab = file.read().split('\n')
            # remove last item if empty
            if len(_vocab[-1]) == 0:
                _vocab = _vocab[::-1]
            _vocabs.extend(_vocab)
    _vocabs = set(_vocabs)

    # merge all vocabs without changing the order of vocabs
    vocabs.extend([v for v in _vocabs if v not in vocabs])

    # save the merged vocabs
    with open(args.output, 'w') as file:
        file.write('\n'.join(vocabs))
    print('Finished writing the merged vocab file.')

if __name__ == '__main__':
    # initialize arg-parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocabs',
                        nargs='+',
                        help='List of vocab.txt files. The indices of vocabs in the first file are unchanged during mergee')
    parser.add_argument('--output',
                        type=str,
                        default='vocab.txt',
                        help='Directory to the save merged vocab file.')
    args = parser.parse_args()

    # execute
    merge(args)
