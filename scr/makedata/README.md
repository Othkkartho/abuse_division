### 데이터셋 수집을 위한 코드들이 있는 폴더

|     파일이름     |                            설명                            |
|:------------:|:--------------------------------------------------------:|
| crawling.py  |              일베 사이트의 게시글 주소를 가져오기 위한 크롤링 코드              |
| crawling2.py |      crawling에서 가져온 게시글 주소를 바탕으로 댓글을 가져오기 위한 크롤링 코드      |
|    eda.py    |          처음 텍스트 위치 변환으로 데이터 증강을 위해 사용을 고려했던 코드           |
| txttocsv.py  | txt로 저장되어 있는 데이터셋을 \|를 기준으로 나눠 text와 label로 csv에 저장하는 코드 |

#### 출처

1. crawling.py, crawling2.py: [Othkkratho - crawling_and_text_maining/crawling.py](https://github.com/Othkkartho/crawling_and_text_maining/blob/master/crawling.py)   
2. eda.py: [catSirup - KorEDA/eda.py](https://github.com/catSirup/KorEDA/blob/master/eda.py)