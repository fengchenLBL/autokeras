"""
test,gs://bucket/filename1.jpeg,daisy
training,gs://bucket/filename2.gif,dandelion
gs://bucket/filename3.png
gs://bucket/filename4.bmp,sunflowers
validation,gs://bucket/filename5.tiff,tulips
"""


import os

prefix = 'gs://autokeras_demo/google_automl/mnist/mnist_png/'
label_train = os.popen('ls training').read().split('\n')[:-1]
label_test = os.popen('ls testing').read().split('\n')[:-1]

png_files_test = [['test,'+prefix+p+','+l for p in os.popen('ls testing/{}/*.png'.format(l)).read().split('\n')[:-1]] for l in label_test]

png_files_train = [['training,'+prefix+p+','+l for p in os.popen('ls training/{}/*.png'.format(l)).read().split('\n')[:-1]] for l in label_train]


import itertools
import pandas as pd

flat_list1 = list(itertools.chain(*png_files_test))
flat_list2 = list(itertools.chain(*png_files_train))
flat_list1.extend(flat_list2)

png_df = pd.DataFrame(flat_list1)

png_df.to_csv('./google_img.csv',index=False, header=False)
