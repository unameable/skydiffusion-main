"""Create an exact-filename paired manifest for G2A-3 History."""

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='datasets/G2A-3')
    parser.add_argument('--output', default='datasets/G2A-3/history_train.csv')
    args = parser.parse_args()
    root = Path(args.data_root).resolve()
    folders = list(root.rglob('SVI_before2014_level20_north1'))
    if len(folders) != 1:
        raise ValueError('Expected exactly one History SVI directory under ' + str(root))
    source_dir = folders[0]
    target_dir = source_dir.parent / 'Aerial_before2014_level20'
    sources = sorted(source_dir.glob('*.jpg'))
    if not sources:
        raise ValueError('No History JPG images found')
    rows = []
    for source in sources:
        target = target_dir / (source.stem + '.png')
        if not target.is_file():
            raise FileNotFoundError(str(target))
        rows.append((source.stem, source.relative_to(root).as_posix(),
                     target.relative_to(root).as_posix()))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['name', 'source', 'target'])
        writer.writerows(rows)
    print('Verified {} pairs; manifest: {}'.format(len(rows), output.resolve()))


if __name__ == '__main__':
    main()
