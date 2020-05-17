---


layout: post
title: "Defending Against Neural Fake News 정리"
excerpt : "Grover 정리"
categories:
  - Papers
tags:
  - Papers
  
  
---


![Grover](/asset/translation/tr1.png)
이 포스트에선 번역했던 논문인 "Defending Neural Fake News" 에 대한 추가적인 설명을 진행하겠습니다.   
해당 포스트의 주제는 NeuIPS 2019 에 accepted 되었던 Defending Neural Fake News에 대한 포스트입니다.   
논문과 더 많은 설명은 [https://rowanzellers.com/grover/](https://rowanzellers.com/grover/) 에서 확인할 수 있습니다.   
   
<h2>1. Grover의 모델 구조 : GPT2</h2>
논문의 __3 Grover: Modeling Conditional Generation of Neural Fake News__에선 
	"We draw on recent progress in training large Transformers for language modeling (Vaswani et al., 2017),   building Grover using the same architecture as for GPT2 (Radford et al., 2019)."
라고, Grover 모델이 GPT2 모델을 그대로 사용한다고 언급하였습니다.   
GPT2 에 대한 설명 이전에, 다른 시계열 모델에 대한 간단한 설명부터 시작하겠습니다.   
RNN 부터 시작된 시계열 모델의 주된 문제는 앞에서 데이터의 정보가 사라지는 Vanishing Gradient Problem이었습니다. 
![wikipedia-LSTM](/asset/explanation/exp2.png)
이를 해결하기 위해 위의 그림과 같은 LSTM을 고안하였습니다. RNN보다 성능 개선이 이뤄졌으나, 여전히 긴 레이어를 가진 LSTM에서 Vanishing Gradient Problem은 해결할 수 없었습니다.   
이 다음으로 Attention이라는 개념이 사용됩니다. 
attention이란 쉽게 설명하면 "Pay Attention!" 과 같은 의미에서처럼 어느 부분에 집중해야 할지 정해주는 개념이라고 생각하면 쉽습니다. 
![wikidocs.net](/asset/explanation/exp3.png)
wikidocs에서 가져온 어텐션에 대한 설명 이미지입니다. 더 자세한 설명은 [wikidocs][https://wikidocs.net/22893] 에서 참고하시기 바랍니다.   
그런데, RNN 모델은 시계열 데이터이기 때문에, 병렬처리가 어렵다는 점에서 Attention과 CNN을 합친 결과가 Transformer 입니다.   
![exp4](/asset/explanation/exp4.png)
"Attention Is All You Need" 에 소개된 Transformer 의 구조입니다.   
Transformer는 다음에 올 토큰을 계산할 때 아래와 같은 수식을 사용합니다. 
![tr2](/asset/translation/tr2.png)
즉, 앞의 N-1 개에 대한 조건부확률을 통해 가장 높은 확률을 가진 토큰을 선택하는 방법입니다. 이런 Transformer의 방식으로 학습한 모델이 바로 GPT2와 BERT 입니다.   다만 BERT는 Bidirectional Encoder Representations from Transformers 라는 뜻으로 bi-directional 하다는 차이가 있습니다. bi-directional 하기 위해서 BERT는 Masking, Next sentence prediction 이라는 두 가지 방법을 이용한 학습을 합니다.   
GPT2의 경우 BERT처럼 bidirectinoal 하지는 않습니다. 따라서 GPT2는 문장의 sequence를 순서대로 생성하고, 생성하면서 또 생성한 데이터를 input으로 사용하며 계속 sequence를 만들어갑니다. (start) 토큰으로 시작해 (end) 토큰이 발생할 때까지 계속 토큰을 생성한다고 보면 될 것 같습니다.   

<h2>2. Top-p sampling (nucleus sampling)</h2>   
본 논문에선 다음에 들어올 토큰을 선택할 때 top-p sampling 을 이용해 문장을 선택하는 방법을 채택하였습니다. NLP 모델에서 빈도를 이용해 단어의 빈도가 가장 높은 단어 순서대로 K개를 선택하고, 이를 확률적으로 나눠 선택하는 Top-k 방식을 선택합니다.   
Top-p 방식은 Nucleus 샘플링이라고도 하며, 토큰의 확률이 가장 높은 순으로 토큰들의 누적합이 p가 될때까지 포함한 단어를 선택하는 방식입니다. Top-k와 다르게, flat distribution에서 top-p는 더 좋은 효과를 거둘 수 있습니다. 위 논문에서 Grover가 더 사람같은 어휘를 구사할 수 있었던 것은 top-k 방식이 아닌 top-p 방식을 사용했기 때문입니다. 또한, peaked distribution에 대해서도, top-p 방식은 확률이 높은 소수만 선택하게 되지만, top-k sampling을 사용할 경우 확률이 매우 낮은 샘플을 어쩔 수 없이 선택하는 문제가 발생할 수 있습니다.   
![top-p](/asset/explanation/exp1.png)
위의 그림은 "The Curious Case of Neural Text Degeneration"의 3.1 Top-p Sampling 에서 가져온 이미지 입니다.

<h2>3. Metadata Set</h2>
Grover는 기사들의 Metadata를 주목했습니다. Grover에선 기사의 여러 요소를 합쳐    
 p(__domain,date,authors,headline,body__)
라는 joint distribution을 만들었습니다.   
이때 각 영역의 시작과 끝에 (start-r), (end-r) 토큰을 삽입했는데,   
예를 들면 authors 토큰의 시작과 끝에는
(start-authors) mr. Journal Ist (end-authors) 라는 토큰을 삽입했다는 뜻입니다.   
따라서, Generator에서 가장 중요한 body를 생성할때는, Grover는 GPT2와 같은 단방향 모델이기 때문에, (start-body) 라는 토큰을 입력하고, (end-body)가 나올때까지 계속 p를 구하면 Grover가 알아서 body를 생성하는 것입니다.   
![article generation](/asset/translation/tr3.png)
또한 위의 그림처럼, context가 완벽하지 않더라도 body를 생성하고, 생성된 body를 이용해 빈 context를 채울 수 있습니다.   

<h2>4. Discrimination </h2>
	(cls)

<h2>5. 왜 Grover가 더 좋을까? </h2>
그렇다면 GPT2와 같은 모델을 사용했는데 왜 Grover가 더 좋은 성능을 도출할까요?   
논문에선 2가지 정도의 원인을 확인할 수 있었습니다. 첫 번째는, openAI에선 training data에 레딧 데이터를 사용하고, Grover는 기사를 바탕으로 학습을 시켰기 때문에 Grover가 더 잘할 수 있는 것이 아닌가 하는 추측입니다.   
두 번째는, 논문에서 주장하는 __exposure bias__ 라는 문제입니다. Generator가 top-p sampling을 통해 분산값을 제한하게 될 경우, Generator가 artifact를 만든다는 것입니다. 여기서 설명하는 artifact란, discriminator가 해당 뉴스가 Generator로 생성된 가짜 뉴스임을 알게 만드는 일종의 단서라고 생각할 수 있습니다.

