### 실제 모델링을 위한 코드들이 모여있는 폴더

|             파일 이름             |                                  설명                                   |                        비고                         |
|:-----------------------------:|:---------------------------------------------------------------------:|:-------------------------------------------------:|
|            cnn.py             |  하이퍼 파라미터 값을 고정하고, 간단한 CNN으로 텍스트 데이터를 학습시켰을 시 어떤 결과가 나오는지 확인하기 위한 코드  |            사용 모델: 1D, Multi-Kernel 1D             |
|   CNN_Classification.ipynb    |                  형태소 분석기를 바꿔 간단히 CNN으로 텍스트 분석을 한 코드                   |                   형태소 분석기: Okt                    |
| CNN_LSTM_Classification.ipynb |                   CNN과 RNN의 LSTM을 이용해 텍스트 분석을 한 코드                    |            형태소 분석기: Mecab, 사용 모델: LSTM            |
|         cnnAutoml.py          |             CNN 모델을 AutoML로 학습시키면 어떤 결과가 나오는지 확인하기 위한 코드              |                 사용한 모델: AutoKeras                 |
|         cnnChange.py          |       CNN에서 하이퍼 파라미터 값들이 어떤 역할을 하는지 바꾸면 어떤 결과가 나오는지를 확인하기 위한 코드       | 변경한 파라미터: embedding_dim, num_filters, batch_size  |
|   LSTM_Classification.ipynb   |                      RNN의 LSTM을 이용해 텍스트 분석을 한 코드                      |             형태소 분석기: Okt, 사용 모델: LSTM             |
|        pretreatment.py        | 데이터셋을 모델 학습할 수 있도록 전처리 해주는 코드. 학습이 끝나고, 테스트를 위해 작성한 문장 분석과 그래프까지 생성함. |            형태소 분석기: Mecab, Max Len: 60            |
|            rnn.py             |  하이퍼 파라미터 값을 고정하고, 간단한 RNN으로 텍스트 데이터를 학습시켰을 시 어떤 결과가 나오는지 확인하기 위한 코드  |                사용 모델: LSTM, BiLSTM                |
|         rnnAutoml.py          |              rnn을 AutoML로 학습시킬 시 어떤 결과가 나오는지 확인하기 위한 코드               |                 사용한 모델: AutoKeras                 |
|         rnnChange.py          |           run에서 하이퍼 파라미터 값을 변경시켰을 때 어떤 변화가 나타나는지 확인하기 위한 코드           | 변경한 파라미터: embedding_dim, hidden_units, batch_size |

#### 출처

1. CNN 학습 코드: [딥러닝을 이용한 자연어 처리 입문 - 1D CNN으로 스팸 메일 분류하기](https://wikidocs.net/80787), [딥러닝을 이용한 자연어 처리 입문 - Multi-Kernel 1D CNN으로 네이버 영화 리뷰 분류하기](https://wikidocs.net/85337)
2. RNN 학습 코드, 전처리: [딥러닝을 이용한 자연어 처리 입문 - 네이버 쇼핑 리뷰 감성 분류하기](https://wikidocs.net/94600), [딥러닝을 이용한 자연어 처리 입문 - BiLSTM으로 한국어 스팀 리뷰 감성 분류하기](https://wikidocs.net/94748)
3. 정확도 그래프: [딥러닝을 이용한 자연어 처리 입문 - 스팸 메일 분류하기](https://wikidocs.net/22894)