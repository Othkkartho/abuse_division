import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from konlpy.tag import Okt
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

data = pd.read_csv('../../data/final.csv', encoding='utf-8')
print('총 샘플의 수 :', len(data))

print(data['text'].nunique(), data['label'].nunique())

print(f'정상 메일의 비율 = {round(data["v1"].value_counts()[0] / len(data) * 100, 3)}%')
print(f'스팸 메일의 비율 = {round(data["v1"].value_counts()[1] / len(data) * 100, 3)}%')
