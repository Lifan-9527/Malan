import malan.preprocessing as prep
import malan
import pandas
import time

def main():

    cache_user = False
    cache_context = True

    if cache_user:
        path = './storage/dataset/train'
        cache_file = './cache/test_cache'

        sub_t = time.time()
        prep.precache_user_map(path, cache_file=cache_file, num_parallel_precache=31)
        print('precache cost: {}'.format(time.time() - sub_t))

        sub_t = time.time()
        res = prep.load_user_map_from_cache(cache_file)
        print('load cost: {}'.format(time.time() - sub_t))
        print(len(res))

    if cache_context:
        source_path = './storage/dataset/train/part_1'
        cache_file = 'cache/map_vid_item_info.pickle'
        prep.precache_item_map(path=source_path, cache_file=cache_file, num_parallel_precache=1)
        item_map = prep.load_item_map(cache_file)
        print(len(item_map))

if __name__ == "__main__":
    main()