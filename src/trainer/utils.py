


def loop(dataset, shuffle: bool = True):
    while True:
        print('\nShuffling dataset ...\n')
        dataset.shuffle()
        for d in range(len(dataset)):
            yield dataset[d]

