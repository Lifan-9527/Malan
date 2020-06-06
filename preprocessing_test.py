import malan.preprocessing as prep
import malan
import pandas
import time

def main():
    path = './storage/dataset/train'
    cache_file = './cache/test_cache'

    sub_t = time.time()
    prep.precache_user_map(path, cache_file=cache_file, num_parallel_precache=31)
    print('precache cost: {}'.format(time.time() - sub_t))

    sub_t = time.time()
    res = prep.load_user_map_from_cache(cache_file)
    print('load cost: {}'.format(time.time() - sub_t))

    print(len(res))


if __name__ == "__main__":
    main()