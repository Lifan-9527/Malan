import malan
from malan import preprocessing
from malan import reader
import traceback

try:

    #path = './storage/dataset/eval/context.parquet'

    path = './storage/dataset/train/part_1/context.parquet'
    d = malan.reader.load_data(path)
    print(d)
except:
    print(traceback.format_exc())


print('ok')