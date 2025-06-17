import zlib
import pickle
import base64
import os

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('thunk_file')
    args = parser.parse_args()
    with open(args.thunk_file, 'r') as f:
        encoded_thunk = f.read()
    thunk = pickle.loads(zlib.decompress(base64.b64decode(encoded_thunk)))
    thunk()