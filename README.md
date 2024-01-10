# Korean-Sentence-Summary-Model
: Language Scoring with Morphological Analysis and Utilization of Perplexity

(형태소 분석 및 Perplexity를 활용한 한국어 텍스트 요약 모델)

_기존에 구현했던 Korean-Sentence-Compression-Model_ver1.0의 문제점을 보완하여 완성하였다._   
_(최종 점수 계산 방식과 문법적 중요도 점수 부여 부분을 수정하였다.)_

■ 프로젝트 설계 배경

- 인터넷에서는 수많은 정보가 쏟아지고 있으며, 사람들은 다양한 정보를 더 빠르고 쉽게 이를 습득하고자 한다. 이에 따라, 긴 영상보다는 1분 내의 짧은 영상을, 긴 글보다는 짧은 글을 선호하게 되었다. 이러한 경향에 맞게, 긴 문서를 간결하게 요약해 주는 텍스트 요약 서비스도 증가하였다.
- 하지만 단순히 텍스트를 짧게 요약하기만 하면, 기존의 정보가 과장되거나 왜곡되어 소비자에게 전달될 수 있다는 위험이 있다. 따라서, **많은 정보가 담긴 문서를 간결하게 요약하되, 문서의 전체 흐름을 정확하게 전달할 필요**가 있다.
- 텍스트 요약 알고리즘 방식
  
  ① 생성 요약 : 핵심 문맥을 반영한 새로운 문장을 생성해서 문서를 요약하는 방식
  
  ② 추출 요약 : 원문에서 중요한 핵심 문장 또는 단어 구를 추출하여 이들로 구성된 요약문을 만드는 방식

- 본 프로젝트에서는,
  
  기존의 ① 생성 요약 방식이, '원문' 뿐만 아니라 '실제 요약문'이라는 label 데이터가 필요하기 때문에 모델을 위한 데이터를 구성하는 것이 힘들고, 데이터를 학습하여 확률상 가장 높은 문장을 생성하기 때문에, 각 문장의 진위를 확인하지 못하여 거짓 정보나 허위 정보를 생성하는 **Hallucination이 발생할 수 있다**는 문제점과,

  ② 추출 요약 방식이, 원문에서 추출된 일부 문장들이 문서 전체 내용을 전부 대변할 수 없기 때문에 내용이 **과장되거나 왜곡되어 전달될 수 있다**는 문제점을 해결하고자 했다.

  "문서의 내용은 다 담으면서 문장의 물리적 길이는 줄여, 왜곡되는 내용과 삭제되는 내용 없이 문서를 읽는 시간 자체를 줄여주는 것"

■ 텍스트 요약 모델 제안

- 본 프로젝트에서는, 기존의 '문장' 추출 요약 방식이 아닌 **'단어' 추출 요약 방식**을 통해 텍스트 요약 모델을 제안한다.
- 문서의 모든 문장들에 대해 각각 요약을 진행하여, **문서의 길이는 짧아지되 문서 전체의 흐름이 반영**되도록 한다.
- 또한, 영어와 달리 어순이 중요하지 않고, 접사, 조사에 따라 문장 성분 및 품사가 달라지는 한국어의 언어적 특성을 고려하여, 한국어 문서에 적용 가능하도록 하였다.
- 입력한 문장에 대해 요약 문장 후보들을 생성하고, 문법적 중요도와 Perplexity를 고려한 최종 점수를 통해 문장 후보 중 최종 요약문을 선택하는 방식으로 진행된다.

![텍스트 요약 모델 구조도_오유솔](https://github.com/YuSol-Oh/Korean-Sentence-Summary-Model_ver2.0/assets/77186075/cdb7b381-e7f3-496f-9b34-596357853bc2)

ⓐ 후보 문장 생성
- 요약할 문서를 구성하는 모든 문장을 하나씩 구분한다.
- 각 문장마다 N-gram 기법을 활용해서 연속된 1 ~ N 개의 크기만큼 연속된 어절들을 반복적으로 삭제하며 다수의 후보 문장을 생성한다.
- 문장마다 삭제된 n개의 어절 자리에는 [MASK]라는 토큰을 채워 넣는다.
- 이때, 요약 문장 후보들이 문장의 주성분인 주어, 목적어, 보어, 서술어는 반드시 포함해야 한다고 판단하여, 후보 문장들의 길이가 단어 4개 미만인 경우에는 후보 문장을 생성하지 않는다.

ⓑ 문장 별 Perplexity 계산
- 생성된 후보 문장이 문맥적으로 자연스러운지 판단하기 위해, KoBERT 데이터 세트로 학습시킨 Masked Language Model을 사용해서 후보 문장마다 Perplexity를 계산한다.
- (Perplexity는 문장의 혼잡도를 의미한다.)

ⓒ 문장 별 문법적 중요도 계산
* 형태소 분석을 활용해 생성된 여러 후보 문장마다 문법적 중요도를 계산한다.
* 후보 문장을 구성하는 각 어절을 확인하면서, KoNLPy에서 제공하는 KOMORAN 형태소 분석기를 이용하여 해당 어절의 형태소 성분을 확인한다.
* 문장의 주성분과 같이 문법적으로 중요한 의미를 갖는 것으로 판단되는 형태소 성분(고유 명사, 일반 명사, 동사, 외국어, 주격 조사, 목적격 조사)에 해당하는 어절을 찾을 때마다 해당 어절에 0.0001씩 중요도 점수를 부여한다.
* 해당 문장에 부여된 모든 중요도 점수를 합산한 값이 해당 문장의 문법적 중요도가 된다.

ⓓ 요약 문서 생성 (최종 점수 계산 및 최종 요약문 선택)
* Perplexity가 낮을수록 문장의 혼잡도가 낮다는 의미이고, 문법적 중요도 값이 클수록 문법적으로 완성도가 높다는 의미이므로, Perplexity의 역수와 문법적 중요도를 곱한 값(최종 점수)이 가장 높은 문장을 원 문장의 최종 요약 문장으로 선정한다.
* 이렇게 문서를 구성하는 모든 문장을 각기 더 짧은 문장으로 요약하고, 이 문장들을 다시 병합하여 최종적으로 요약된 문서를 완성한다.

![image](https://github.com/YuSol-Oh/Korean-Sentence-Summary-Model_ver2.0/assets/77186075/4054aacd-b0f9-46da-ba10-c4a9dcf492bd)

■ 텍스트 요약 모델 평가

- 텍스트 요약 모델 성능 평가를 위해, 사건 단어 주의 집중 메커니즘을 활용해서 문장 요약을 시도한 연구에서 사용된 2,865개의 원본 문서와 정답 요약 문서 쌍으로 구성된 데이터 세트를 사용하였다.

① 코사인 유사도
* 요약된 문서가 원본 문서의 핵심 어절을 포함하고 있는지 평가하기 위해, 모델이 생성한 요약문과 정답 요약문의 유사 정도를 평가하였다.
* 전체 데이터에 대해서 코사인 유사도를 계산 후 평균을 내었을 때, 약 0.68의 유사도를 보였다.

```python
# 파일 경로 설정
source_file = './source_축약 전 원문 파일.txt'
target_file = './target_정답 축약 문장 파일.txt'

source_sentences = []
target_sentences = []

with open(source_file,'r',encoding='utf-8') as r:
  source_sentences = r.readlines()

with open(target_file, 'r', encoding='utf-8') as r :
  target_sentences = r.readlines()

# 각 경우의 수에 대해 문장 압축을 수행하고 결과 저장
all_cosine_similarities = []

for i in range(len(source_sentences)): # 각 문장 쌍에 대해서

  input_sentence = source_sentences[i].strip()
  target_sentence = target_sentences[i].strip()

  print(int(time.time()-start_time), "초 소요", datetime.datetime.now())

  compressed_sentence = compress_sentence(input_sentence, a, b, c, d, e, f, g) # 문장 요약
  target_embedding = encode_sentence(target_sentence ) # 정답 요약 문장
  compressed_embedding = encode_sentence(compressed_sentence) # 모델 요약 문장
  similarity = cosine_similarity(target_embedding, compressed_embedding)[0][0] # 유사도 계산

  # 현재 문장 쌍에 대해 코사인 유사도, parameter를 리스트에 추가
  all_cosine_similarities.append(similarity)
```

② ROUGE-W 점수
* 모델이 생성한 요약문과 정답 요약문 사이의 연속된 일치를 고려하여 가중치를 할당한 점수이다.
* ROUGE-W 점수가 작을 수록 두 문장이 유사함을 의미한다.
* 전체 데이터의 ROUGE-W 점수 분포를 보았을 때, 0에 가깝게 분포함을 확인할 수 있었다.

```python
def f(k):
    return k**0.5

def rouge_lcs(target, model, weighted=True):
    m, n = len(target), len(model)

    # Initialize the c-table
    c_table = [[0]*(n+1) for i in range(m+1)]
    # Initialize the w-table
    w_table = [[0]*(n+1) for i in range(m+1)]

    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0:
                continue
            # The length of consecutive matches at
            # position i-1 and j-1
            elif target[i-1] == model[j-1]:
                # Increment would be +1 for normal LCS
                k = w_table[i-1][j-1]
                increment = f(k+1) - f(k) if weighted else 1
                # Add the increment
                c_table[i][j] = c_table[i-1][j-1] + increment
                w_table[i][j] = k + 1
            else:
                if c_table[i-1][j] > c_table[i][j-1]:
                    c_table[i][j] = c_table[i-1][j]
                    w_table[i][j] = 0 # no match at i,j
                else:
                    c_table[i][j] = c_table[i][j-1]
                    w_table[i][j] = 0 # no match at i,j
    return c_table[m][n]
```

③ 네이버 CLOVA Summary API와 비교
* 본 프로젝트에서 제안한 단어 추출 요약 모델을 이용했을 때와, 네이버의 문장 추출 요약 모델을 이용했을 때의 요약 결과를 비교하였다.
* 원문 기사의 제목을 보면, 사건의 현상에 대한 이유를 전달하는 것이 이 기사의 목표임을 알 수 있다. 그러나 네이버 기사 요약 서비스가 생성한 요약문은 AI가 판단했을 때 중요하다고 생각되는 문장 3개만을 추출하기 때문에, 현상에 관한 내용만을 포함하고 원인에 관한 내용은 포함하지 않고 있다. 반면, 제안 모델을 통해 요약된 문서는 현상의 원인에 대한 핵심 문장을 모두 포함하여 기존 글의 전체적인 흐름을 잘 대변하고 있음을 확인할 수 있다.

![image](https://github.com/YuSol-Oh/Korean-Sentence-Summary-Model_ver2.0/assets/77186075/6fcd456d-f587-4119-b330-92cfc70eb8e2)

■ 결론

* 본 프로젝트에서는 단어 추출 방식을 기반으로 한국어의 문법적 특성과 문장 내 단어 간의 자연스러운 정도를 고려하며, 삭제되는 문장 없이 문서의 모든 흐름을 담을 수 있는 요약 모델을 제안하였다.
* 제안 모델을 통해 생성된 요약문을 보았을 때, 허위 정보를 생성할 가능성이 없고, 문서의 모든 흐름이 담긴다는 결과를 보였다.
* 하지만, 문장 간의 연결이 부자연스러운 경우가 존재하고, 어절의 삭제로 인해 각 문장 내에서 문법적, 문맥적 완성도가 떨어지는 경우가 있음을 확인하였고, 개체명 인식에 약하다는 한계를 보였다.
* 위의 문제점을 해결하기 위해, LLM을 사용해보기, 추출한 단어들로 문장 재생성하기, 국립국어원이 공개한 개체명 데이터를 활용하여 개체를 인식할 수 있도록 수정하기 등의 방법을 고려해볼 수 있었고, 추후의 연구로 진행할 예정이다.

■ 코드

(google colab을 이용)
* install
```python
!pip install konlpy
!pip install mxnet
!pip install gluonnlp pandas tqdm
!pip install sentencepiece
!pip install transformers
!pip install torch
!pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'
```
* 형태소 분석기 및 KoBERT 데이터 세트로 학습시킨 Masked Language Model
```python
from konlpy.tag import Komoran
komoran = Komoran()

from transformers import AutoTokenizer, AutoModelForMaskedLM
tokenizer = AutoTokenizer.from_pretrained("monologg/kobert")
model = AutoModelForMaskedLM.from_pretrained("monologg/kobert")
```
* import
```python
import re
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from konlpy.tag import Komoran
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import itertools
import numpy as np
import time, datetime
import matplotlib.pyplot as plt
```
ⓐ 후보 문장 생성 → ⓑ 문장 별 Perplexity 계산 → ⓒ 문장 별 문법적 중요도 계산 → ⓓ 요약 문서 생성 (최종 점수 계산 및 최종 요약문 선택)
```python
# 문단을 문장으로 나누는 함수
def sentence_tokenizer(paragraph):
    return re.split(r'(?<=[.!?])\s+', paragraph)

# 문장 어절 단위로 토큰화
def word_tokenizer(sentence):
    return sentence.split(' ')

# 따옴표와 따옴표를 포함하는 문장을 하나의 토큰으로
def process_quoted_words(tokens):
    processed_tokens = []
    quoted_word = ""
    in_quote = False

    for token in tokens:
        if "'" in token:
            if in_quote:
                quoted_word += " " + token
                processed_tokens.append(quoted_word)
                quoted_word = ""
                in_quote = False
            else:
                quoted_word = token
                in_quote = True
        else:
            if in_quote:
                quoted_word += " " + token
            else:
                processed_tokens.append(token)
    return processed_tokens
```
```python
# 형태소 분석 후 문법적 중요도 점수 추가
def add_linguistic_score(sentence):
    pos = komoran.pos(sentence)
    score = 0
    for p in pos:
        if p[1] == 'NNP':
            score += 0.000001
        elif p[1] == 'NNG':
            score += 0.000001
        elif p[1] == 'vv':
            score += 0.000001
        elif p[1] == 'SL':
            score += 0.000001
        elif p[1] == 'JKS':
            score += 00.000001
        elif p[1] == 'JKO':
            score += 0.000001
    quoted_words = re.findall(r"'(.*?)'", sentence)
    for quoted_word in quoted_words:
        score += 0.0001
    return score

# KoBERT 모델을 이용하여 perplexity 계산
def calculate_perplexity_score(sentence):
    # KoBERT 모델이 이해할 수 있는 형태로 tokenize
    tokens = tokenizer.tokenize(sentence)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    # [MASK] 토큰 위치 찾기
    mask_token_index = token_ids.index(tokenizer.mask_token_id)

    # 입력값 생성
    input_ids = torch.tensor([token_ids])
    outputs = model(input_ids)
    predictions = outputs[0]

    # [MASK] 토큰 위치에 대한 예측값 추출
    masked_predictions = predictions[0, mask_token_index]

    # softmax 함수를 이용하여 확률값을 확률 분포로 변환
    probs = torch.softmax(masked_predictions, dim=-1)

    # perplexity 계산
    perplexity = torch.exp(torch.mean(torch.log(probs)))

    return perplexity.item()
```
```python
# 압축 문장 후보 생성 및 문법적 중요도 점수 계산
def compress_sentence(token):
    compressed_candidates = []
    max_n = 4 #len(token) - 4

    for n in range(1, max_n + 1):
        for i in range(len(token) - n + 1):
            compressed_tokens = token[:i] + token[i + n:]
            compressed_sentence = " ".join(compressed_tokens)
            score = add_linguistic_score(compressed_sentence)
            compressed_candidates.append((compressed_sentence, score, n))

    perplexity_scores = []
    compressed_candidates_with_score = []

    for n in range(1, max_n + 1):
        for i in range(len(token) - n + 1):
            mask_idx = list(range(i, i + n))
            masked_tokens = list(token)
            for j in mask_idx:
                masked_tokens[j] = "[MASK]"
            masked_sentence = " ".join(masked_tokens)

            perplexity_score = calculate_perplexity_score(masked_sentence)
            linguistic_score = compressed_candidates[i][1]
            final_score = perplexity_score - linguistic_score

            perplexity_scores.append(perplexity_score)
            compressed_candidates_with_score.append((re.sub(r'\[MASK\]\s*', '', masked_sentence), final_score, n))

    compressed_candidates_with_score_sorted = sorted(compressed_candidates_with_score, key=lambda x: x[1])
    final_compressed_sentence = re.sub(r'\[MASK\]\s*', '', compressed_candidates_with_score_sorted[0][0])
    selected_n = compressed_candidates_with_score_sorted[0][2]

    return compressed_candidates_with_score_sorted, final_compressed_sentence, selected_n
```
```python
# 문단을 한 문장씩 나눠서 압축하고, 최종 결과를 한 문단으로 합치는 함수
def compress_paragraph(paragraph):
    sentences = sentence_tokenizer(paragraph)
    compressed_sentences = []

    for sentence in sentences:
        token = word_tokenizer(sentence)
        token = process_quoted_words(token)
        compressed_candidates_with_score_sorted, final_compressed_sentence, selected_n = compress_sentence(token)
        compressed_sentences.append(final_compressed_sentence)

    compressed_paragraph = ' '.join(compressed_sentences)
    return compressed_paragraph

# 입력 문단
paragraph = input()

# 문단을 압축한 결과
compressed_paragraph = compress_paragraph(paragraph)

# 결과 출력
print("입력 문단:")
print(paragraph)

print("\n최종 요 문단:")
print(compressed_paragraph)
```
