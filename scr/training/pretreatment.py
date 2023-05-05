import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from konlpy.tag import Mecab
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def pretreatment():
    # plt 폰트 설정
    plt.rcParams["font.family"] = 'NanumGothic'

    data = pd.read_csv('../../data/final.csv', encoding='utf-8')
    # print('총 샘플의 수 :', len(data))

    # print(data['text'].nunique(), data['label'].nunique())

    # print(f'욕설이 아닌 댓글의 비율 = {round(data["label"].value_counts()[0] / len(data) * 100, 3)}%')
    # print(f'욕설인 댓글의 비율 = {round(data["label"].value_counts()[1] / len(data) * 100, 3)}%')

    # print(data.isnull().values.any())  # Null 값 유무 확인

    # 훈련 데이터와 테스트 데이터 분리 8:2로 분리
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    # print('훈련용 댓글의 개수 :', len(train_data))
    # print('테스트용 댓글의 개수 :', len(test_data))

    # 레이블 분포 확인
    train_data['label'].value_counts().plot(kind='bar')

    # print(train_data.groupby('label').size().reset_index(name='count'))

    # 데이터 정제하기

    # 한글과 공백을 제외하고 모두 제거
    train_data['text'] = train_data['text'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
    train_data['text'].replace('', np.nan, inplace=True)
    # print(train_data.isnull().sum())

    test_data.drop_duplicates(subset=['text'], inplace=True)  # 중복 제거
    test_data['text'] = test_data['text'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")  # 정규 표현식 수행
    test_data['text'].replace('', np.nan, inplace=True)  # 공백은 Null 값으로 변경
    test_data = test_data.dropna(how='any')  # Null 값 제거
    # print('전처리 후 테스트용 샘플의 개수 :', len(test_data))

    # 불용어 정의
    stopwords = ['도', '는', '다', '의', '가', '이', '은', '한', '에', '하', '고', '을', '를', '인', '듯', '과', '와', '네', '들', '듯',
                 '지', '임', '게', '만', '게임', '겜', '되', '음', '면']

    # 토큰화

    mecab = Mecab("C:/mecab/mecab-ko-dic")

    train_data = train_data.dropna(subset=['text'])
    train_data['tokenized'] = train_data['text'].apply(mecab.morphs)
    train_data['tokenized'] = train_data['tokenized'].apply(lambda x: [item for item in x if item not in stopwords])
    test_data['tokenized'] = test_data['text'].apply(mecab.morphs)
    test_data['tokenized'] = test_data['tokenized'].apply(lambda x: [item for item in x if item not in stopwords])

    X_train = train_data['tokenized'].values
    y_train = train_data['label'].apply(lambda x: int(x)).values
    X_test = test_data['tokenized'].values
    y_test = test_data['label'].apply(lambda x: int(x)).values

    # 정수 인코딩

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)

    threshold = 2
    total_cnt = len(tokenizer.word_index)  # 단어의 수
    rare_cnt = 0  # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
    total_freq = 0  # 훈련 데이터의 전체 단어 빈도수 총 합
    rare_freq = 0  # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

    # 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
    for key, value in tokenizer.word_counts.items():
        total_freq = total_freq + value

        # 단어의 등장 빈도수가 threshold보다 작으면
        if (value < threshold):
            rare_cnt = rare_cnt + 1
            rare_freq = rare_freq + value

    # print('단어 집합(vocabulary)의 크기 :', total_cnt)
    # print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s' % (threshold - 1, rare_cnt))
    # print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt) * 100)
    # print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq) * 100)

    # 전체 단어 개수 중 빈도수 2이하인 단어 개수는 제거.
    # 0번 패딩 토큰과 1번 OOV 토큰을 고려하여 +2
    vocab_size = total_cnt - rare_cnt + 2
    # print('단어 집합의 크기 :', vocab_size)

    tokenizer = Tokenizer(vocab_size, oov_token='OOV')
    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    # print(X_train[:3])
    # print(X_test[:3])

    # 패딩

    print('댓글의 최대 길이 :', max(len(review) for review in X_train))
    print('댓글의 평균 길이 :', sum(map(len, X_train)) / len(X_train))
    plt.hist([len(review) for review in X_train], bins=50)
    plt.xlabel('length of samples')
    plt.ylabel('number of samples')
    # plt.show()

    def below_threshold_len(max_len, nested_list):
        count = 0
        for sentence in nested_list:
            if (len(sentence) <= max_len):
                count = count + 1
        # print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s' % (max_len, (count / len(nested_list)) * 100))

    max_len = 60
    below_threshold_len(max_len, X_train)

    X_train = pad_sequences(X_train, maxlen=max_len)
    X_test = pad_sequences(X_test, maxlen=max_len)

    return X_train, X_test, y_train, y_test, vocab_size, max_len, stopwords, tokenizer


def sentiment_predict(new_sentence, loaded_model, stopwords, tokenizer, max_len):
    mecab = Mecab("C:/mecab/mecab-ko-dic")

    new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]', '', new_sentence)
    new_sentence = mecab.morphs(new_sentence)  # 토큰화
    new_sentence = [word for word in new_sentence if not word in stopwords]  # 불용어 제거
    encoded = tokenizer.texts_to_sequences([new_sentence])  # 정수 인코딩
    pad_new = pad_sequences(encoded, maxlen=max_len)  # 패딩
    score = float(loaded_model.predict(pad_new))  # 예측
    if (score > 0.5):
        print("{:.2f}% 확률로 옥설입니다.".format(score * 100))
    else:
        print("{:.2f}% 확률로 욕설이 아닙니다.".format((1 - score) * 100))
