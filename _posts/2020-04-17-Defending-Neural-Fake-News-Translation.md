---


layout: post
title: "Defending Against Neural Fake News"
excerpt : "논문 소개 및 번역"
categories:
  - Papers
tags:
  - Papers
  
  
---
![tr1.png](/_asset/image/translation/tr1.png)
NeurIPS 2019에 채택된 Defending Against Neural Fake News 입니다.
<href>https://rowanzellers.com/grover/</href> 를 통해 더 많은 정보와 코드를 볼 수 있습니다.

직접적인 설명 이전에 한글 번역으로 시작해보려 합니다.
해당 논문의 마지막에는 신경망으로 생성될 Fake News에 대한 연구 윤리까지 제시하고 있습니다. 이 부분은 생략하겠습니다.
Appendix에 나와있는 모델 구조는 추후에 다시 포스팅하겠습니다.
오역 및 미숙한 이해 부분은 차차 개선하겠습니다. 현재 오역의 여지가 있는 부분은 *으로 표시하였습니다.

0. 초록
최근의 자연어 처리는 두 가지 측면을 보입니다. 요약이나 번역 등에서 긍정적인 모습을 보이는 반면, neural fake news를 만드려는 부정적인 측면 또한 가지고 있습니다. 여기서 neural fake news란, 실제 뉴스를 매우 그럴싸하게 흉내내는 특정 목적을 가진 가짜 뉴스를 의미합니다.
현대 컴퓨터 보안은 threat modeling 에 의존하고 있습니다. 이는 공격자의 시각으로 보안의 취약점을 찾아내는 것을 의미합니다. 즉, neural fake news 를 잘 대응하기 위해선 먼저 그 대상을 조사하고 위험성을 특징짓는 것이 중요합니다. 따라서, 우리는 주제를 선택가능한 기사 생성기 Grover를 제시합니다. Grover는 '백신과 자폐의 연관성이 발견되다' 라는 기사 제목을 작성하면 나머지 부분을 스스로 작성할 수 있습니다. 사람들은 Grover가 작성한 기사에 사람이 쓴 가짜 뉴스보다 더 높은 신뢰성을 가졌습니다.
Grover같은 가짜 뉴스 생성기에 대항하는 입증 기술을 만드는 것이 매우 중요합니다. 훈련 데이터에 접근할 수 있을 때, 우리는 가장 성능이 좋은 discriminator들이 73%의 정확도로 사람이 쓴 기사와 신경망으로 생성된 기사를 구분함을 알 수 있었습니다. 납득이 어렵겠지만, Grover의 가짜 기사를 가장 잘 구분할 수 있는 것은 Grover 자체라는 결과가 나왔습니다. 더 나아가, 노출 편향 (exposure bias)이 Generator와 비슷한 discriminator가 이용가능한 요소들을 만들어낸다는 것을 알아낼 수 있었습니다.
마지막으로, 기술에 대한 윤리 문제의 논의 끝에 신경망 생성 가짜 뉴스 탐색의 발전을 위해 Grover를 배포하기로 결정했습니다.

1. 도입
온라인에서 퍼지는 가짜 뉴스들은 최근 주된 사회 문제로 부상했습니다. 광고 수익, 의견 주장, 심지어 선거를 이기기 위해서까지 가짜 구설수들을 만들어 퍼트리고 있습니다. 따라서 인터넷에서 이런 거짓 정보들이 퍼지는 것을 막는 것이 기술적, 정치적으로 중요한 이슈가 되고 있습니다.

현재 알려진 바로는 대부분의 가짜 정보들은 사람들이 임의로 만든 것입니다. 그러나, 자연어 생성 기술이 발달하면서, 이를 이용해 진짜 같아 보이는 선전물들을 자신이 원하는 대로 만들어 배포할 것입니다. 따라서, 우리가 텍스트 생성 기술의 진보에 감탄하면서도, 한편으론 AI 기반의 '신경망으로 제작된' 가짜 뉴스에 대해 우려할 필요가 있습니다.

이 논문에선 실제 이런 일들이 퍼지기 전에 이 현상을 이해하고 대응해보려 합니다. 즉, 보안 방식의 일종인 threat modeling을 이용할 것입니다. 이는 방어 대책을 마련하기 위해서 발생 가능한 위협과 취약점을 분석하는 행위를 뜻합니다. 허위 정보의 위험성 연구를 위해, 이를 생성하는 모델 Grover를 제시합니다. Grover는 완전히 새로운 기사를 만들 수 있는데, 또한 주제를 조종할 수 있습니다. 텍스트 내용 뿐만 아니라 제목, 출처, 날짜, 기자명까지 말이죠. Grover는 주제를 조종할 수 있는 텍스트 생성에 대한 adversary 역할을 할 것입니다.

사람들은 사람이 직접 쓴 허위 정보보다 Grover로 만든 허위 정보가 더 믿을만하다고 평가했습니다. 따라서, Grover와 같은 생성자들에 대한 정보 입증 기술을 발전시키는 것은 매우 중요한 연구 주제입니다. 우리는 구별자가 5000개의 Grover가 생성한 기사를 참고하고, 실제 뉴스 기사는 제한 없이 참조할 수 있다고 가정했습니다. 이 환경 구성에서 가장 우수한 deep pretrained language models은 73%를 달성하였습니다. 그러나, Grover 가 discriminator로 작동할때, 92%의 성능을 내는 것을 확인할 수 있었습니다. 즉, 가짜 뉴스를 생성하는 모델이 가짜 뉴스를 가장 잘 구분한다는 것이었습니다.

또한 우리는 어떻게 deep pretrained language model들이 실제 뉴스와 가짜 뉴스를 구분하는지 연구했습니다. 이때, 뉴스 생성중에 주된 요소가 노출 편향(exposure bias)로써 생성되는 것을 알 수 있었습니다. 즉, 생성자는 완벽하지 않기에, 기사 길이가 늘어나며 out-of-distribution(비정상 샘플) 을 더 떨어트리는 분포에서 샘플링을 진행합니다.
그러나, 이러한 효과를 완화하는 이 sampling 전략은 또한 discriminator가 이용할 수 있는 강력한 도구를 제공합니다.

*(Policy 부분 문단 생략하였습니다.)

2. Fake News in a Neural and Adversarial Setting
사람들이 작성한 가짜 뉴스가 다양하기 때문에 adversary가 무엇을 해야할지, 그리고 이를 확인하는 verifier는 무엇을 해야할지 제시하고자 합니다.

가짜 뉴스의 범위.
가짜 뉴스는 풍자부터 선동까지 그 범주가 넓습니다. 이 논문에서는 텍스트로만 구성된 뉴스 기사 형식의 문서로 제한합니다. 현재의 가짜 뉴스들은 대부분 사람이 직접 작성하였으며, 클릭을 통한 광고 수익이나, 전파하고자 하는 정보를 알리려는 선전용 기사가 존재합니다. 이 기사들은 adversary로 하여금 주어진 제안이나, 바이럴 컨텐츠만을 제공해야 할지 선택하게 합니다. *

팩트 체크와 입증 ( 관련 연구 )
온라인 허위 사실을 맞서려는 상당한 연구가 있었습니다. 페이스북의 경우 믿을만한 소스들을 우선시하고, 허위 사실을 유포하는 계정을 차단하였습니다. 혹은 유저들이 NewsGuard, Hoaxy같은 툴을 사용해 가짜 뉴스를 피하는 경우, Snopes,PolitiFact같은 웹사이트를 사용하는 경우도 있었습니다. 이 서비스들은 주장과 기사, 전체 웹사이트를 분석하려는 수동적인 노력에 의존합니다. 자동으로 만들어진 가짜 뉴스를 찾는데는 주로 글 스타일의 편향성을 이용합니다. 이런 노력들은 의심스러운 계정들을 차단하는데 도움이 되었습니다. 그러나, 팩트 체킹은 만병통치약은 아닙니다. 역효과 현상(backfire effect)이나 확증 편향(confirmation bias)와 같은 인지 편향(cognitive biases) 은 사람들이 자신이 믿는 성향의 뉴스를 더 잘 믿게 만듭니다.

프레임워크
본 논문은 가짜 뉴스 생성과 감지를 하나의 adversarial game으로 간주하고, 게임의 두 player를 제시합니다.
adversary (적대자)
adversary의 주 목적은 특정 속성을 만족하는 가짜 스토리 생성입니다. 스토리는 사람과 verifier 모두에게 진짜처럼 보여야 합니다.
verifier (입증자)
verifier의 목적은 기사 내용이 진짜인지 거짓인지 구분입니다. 입증자는 진짜 뉴스에는 제한이 없지만 적대자로부터 일부의 기사만을 받습니다. 이런 설정은 현 상황을 그대로 반영합니다. 즉, 한 플랫폼이 불량 계정을 차단시킬때, 계정이 배포하던 가짜 뉴스들을 학습 데이터로 반영가능하기 때문입니다. 반면, 새로 생성된 계정에 대해선 가짜 뉴스를 수집하기 어렵습니다.

두 플레이어들은 공격자와 방어자로써 경쟁을 벌입니다. 입증자가 치밀해질수록 적대자도 치밀해집니다. 따라서, 우리는 다음 섹션에서 다룰, 가장 강한 적대자의 공격에 대해 준비할 필요가 있습니다.

3. Grover: Modeling Conditional Generation of Neural Fake News
현재 온라인 상의 허위 정보를 보면, 적대자들은 특정 목적을 가진 컨텐츠를 만든다고 가정가능합니다. Radford의 19년 논문을 통해 최근 논문들이 실제 사람이 쓴 것 같은 텍스트를 만들 수 있음이 알려졌지만, 사람이 주제를 조정 가능한 텍스트 생성은 없었습니다. 그러므로, 가짜 뉴스의 실제 특성을 반영하기 위해서는 Grover는 실제 기사같으면서도 목적을 조종할 수 있는 생성이 가능해야 합니다.

문서 X의 확률은 이전 토큰들이 생성되었을 때 Xi번째 토큰이 생성될 때의 조건부 확률의 곱 입니다.
문서는 구조화되지 않은 text field로 취급되는데, <start> 토큰으로 시작하고, <end> 토큰으로 종료합니다. <end>는 텍스트 생성을 끝내기 때문에 특히 중요합니다. 그러나, 뉴스 기사는 text field 외에 다른 구조도 필요로 합니다. Metadata field는 해당 기사가 작성된 웹사이트인 domain, 기사 작성 날자인 date, 저자인 authors, 그리고 headline을 포함합니다. 뉴스 기사를 만든다는 것은 이 모든 요소들을 필요로 할 뿐만 아니라, 이 요소들을 이용해 기사 생성을 조종할 수 있습니다. 하나의 기사는 따라서 다음의 결합 분포를 따릅니다.

p(domain, date, authors, headline, body)  - (Equation 1)

아직 이 식으로부터 어떻게 샘플을 만들지 명확하지 않습니다. 하나의 방법은 article's field를 canonical order로 정의하는 것입니다. 그 뒤 order 순서대로 model을 구성합니다. 그러나 이런 ordering 방식은 너무 비용이 큰 marginalization 없이는 sampling을 금지합니다. * 

Grover의 접근법은 multi-field를 가진 문서들을 효율적으로 생성하는 새로운 접근법을 사용합니다. Grover는 Equation 1의 모델 프레임워크가 Equation 2로 유연하게 분해되도록 하는 방식을 채택했습니다. inference를 하는 동안 f의 집합 field F로 시작합니다. 이때 f는 field-specific한 start, end 토큰을 가지고 있습니다. 우리는 field를 (domain,date,authors,headlines,body) 순으로 정렬하고 resulting token을 합쳤습니다. 생성해야하는 target인 target field를 만들기 위해서, start token <strat-r> 로 추가했습니다. 그리고 <end-r>에 닿을때까지 모델을 시도합니다.

Figure 2는 어떻게 Grover가 백신을 반대하는 기사를 만드는지의 일련의 과정입니다. 적대자가 domain, date, headline을 명시받습니다. 그 뒤 body를 만들고, 이는 다시 author와 headline을 만드는데 사용가능합니다.

모델 훈련동안, 랜덤하게 나뉜 F1, F2의 공동의 셋을 이용해 추론을 합니다.* 또한 무작위로 각각의 field를 10%의 확률로 drop 합니다. 그러나, body의 경우 35%의 확률로 drop 합니다. 이 과정이 모델이 unconditional generation을 수행하도록 만듭니다. 앞서 언급한 순서대로 field를 정렬한 뒤, 토큰을 합칩니다. 모델은 그 뒤 F1의 cross-entropy를 줄이는 방향으로 훈련됩니다.

Architecture
언어 모델링을 위해, Transformer를 이용하는데, Grover는 GPT2와 같은 구조를 가지도록 설계하였습니다. 우리는 3가지 모델 사이즈를 고려합니다. Grover Base는 가장 작은 모델로, GPT, BERT-Base와 같이 12개의 layer와 124백만 개의 parameter를 가지도록 만듭니다. 다음 모델은 Grover Large로, 24개의 Layer와 355 백만 개의 파라미터를 가지는데, 이는 BERT-Large와 같습니다. 마지막으로 Grover Mega는 48개의 Layer와 15억개의 파라미터를 가지는데 이는 GPT2와 동일합니다.

Dataset
우리는 RealNews라는 Common Crawl에서 가져온 뉴스 기사 코퍼스를 제시합니다. Grover는 metadata를 가진 매우 큰 양의 뉴스 기사 코퍼스가 필요합니다. 물론 현재 상응하는 데이터는 없습니다. 따라서 우리는 Common Crawl에서 모아서 만드는데, Google News에서 5000개의 뉴스 주소만을 사용해 모았습니다. Newspaper Python Library를 사용해 body 필드와 metadata를 나누는 데에 사용했습니다. Common Crawl의 뉴스들은 2016년 12월부터 2019년 3월까지의 뉴스를 훈련 데이터로, 2019년 4월 데이터는 evaluation에 사용하였습니다. 중복 기사들을 제거하고 나서, RealNews는 120 GB정도 크기를 갖게 되었습니다.

Learning 
Grover 모델은 RealNews 데이터의 무작위로 샘플링된 1024 시퀀스로 훈련시켰습니다. 다른 hyperparameter는 appendix를 참고하시기 바랍니다. iteration은 80만번이고, 512 batch size에 256 TPU v3 cores를 사용했습니다. 훈련 시간은 2주가 걸렸습니다.

3.1 Language Modeling Results : Measuring the importance of data, context, and size
Grover는 2019년 4월 테스트 셋을 이용해서 validation을 진행했습니다. 이때 다른 unconditional language model과 비교했습니다. 두 가지의 evaluation을 진행합니다.
	Unconditional: context가 주어지지 않고 모델이 body를 생성해야 하는 점.
	conditional: 완전한 metadata가 주어지는 점.
이때 body만을 이용해 perplexity를 측정합니다.

결과는 Figure 3에서 확인할 수 있습니다. Grover는 conditional validation에서 눈에 띌 정도로 Perplexity를 낮췄습니다. 또한 Grover의 사이즈가 큰 모델을 쓸수록 perplexity가 떨어졌습니다. 아마도 OpenAI WebText 코퍼스가 뉴스가 아닌 기사들도 포함하고 있기 때문에 GPT2모다 높은 성능을 낸다고 생각됩니다.

3.2 Carefully restricting the variance of generations with Nucleus Sampling
Grover로 샘플링 하는 것은 모델이 left-to-right 언어 모델처럼 진행하기에 복잡하지 않아 보입니다.
그러나 디코딩 알고리즘을 잘 고르는 것이 중요합니다. 최대가능도(likelihood-maximization) 전략이 closed-ended 생성에서 잘 동작하였지만, open-ended generation에서 degenerate text를 만드는 것을 확인할 수 있었습니다.* 그러나, 6장에서 볼 수 있듯이, 분산값을 제한하는 것 또한 중요합니다.

이 논문에서는 Nucleus Sampling을 사용합니다. 제한값 p가 주어질때, 각 스텝마다 누적 확률이 top p%에 속하는 단어들로 샘플링을 합니다.

4. Humans are Easily Fooled by Grover-written Propaganda
가장 큰 모델인 Grover-mega로 만든 허위 정보를 평가했습니다. 제한값 p=0.96입니다. 4 종류의 기사를 함께 준비합니다. 사람이 쓴 저명한 뉴스 기사 (Human News), Grover가 해당 Human News를 이용해 쓴 (Machine News), 가짜 뉴스 웹사이트의 기사(Human Propaganda), 그리고 Graver가 해당 Human Propaganda를 이용해 쓴 (Machine Propaganda) 가 그 4 종류의 기사입니다. Amazon Mechanical Turk를 이용해 세 가지를 평가했습니다. 
	스타일의 일관성
	내용의 합리성
	전체적인 신뢰도

그 결과 Grover 가 쓴 기사가 사람이 쓴 기사만큼 신뢰도가 높진 않지만 propaganda보다는 높음을 확인할 수 있었습니다. 또한 Machine Propaganda 도 Human Propaganda보다 신뢰도가 높았습니다.

5. Neural Fake News Detection

Grover로 만들어진 가짜 뉴스들은 neural fake news detection이 중요한 연구 주제임을 밝혀냈습니다. 입증자를 이용하면 이를 구분할 수 있습니다. 
a. Grover. 텍스트 생성 대신 구별을 할 수 있는 버전을 고려했습니다. GPT와 유사하게, [CLS]라는 특별한 토큰을 각 기사의 끝마다 삽입하고, final hidden state에서 이를 추출합니다. hidden state는 linear layer로 학습되어* Human과 Machine을 구분하게 됩니다. 
To simulate real conditions, and ensure minimal overlap between the generator and discriminator
parameters, we initialize Grover for discrimination using the checkpoint at iteration 700k, whereas
the generator uses the checkpoint at iteration 800k

b. GPT2
c. BERT
d. FastText

모든 모델들은 cross entropy를 최소화 하는 방향으로 훈련되었습니다.

5.1 A semi-supervised setting for neural fake news detection
온라인엔 매우 많은 기사들이 있지만, 대부분 검색되는 기사들은 최근 일들을 다루고 있습니다. 따라서 적대자로부터 생성된 가짜 뉴스가 매우 적을 수 있습니다. 즉 fake news detection을 semi-supervised problem으로 정의합니다. 신경망의 입증자는 RealNews 트레이닝 셋의 모든 사람이 쓴 기사들에 접근할 수 있습니다. 그러나 적대자로부터 생성된 기사에는 한계가 있습니다. 2019년 4월의 뉴스를 이용해 body text 1만개를 만듭니다. 다른 1만개는 사람이 작성한 뉴스로 사용합니다. 이 2만개의 데이터를 잘 분리하여 1만개는 학습, 2천개는 validation, 8천개는 테스트에 사용합니다.
두가지의 평가 방식을 고려합니다. Unpaired 세팅에선 구별자는 단 하나의 뉴스를 입력받으며 Human, Machine을 구분해야 합니다. paired 세팅에선 같은 metadata를 가지는 뉴스 기사 2개를 받으며, 하나는 실제 기사이고 하나는 신경망으로 만들어진 기사입니다. 이때, 구별자는 신경망으로 만들어진 기사에 Machine 일 확률을 더 높게 부여해야 합니다. 

5.2 Discrimination results: Grover performs best at detecting Grover’s fake news
실험 결과는 몇 가지 특성을 보여줍니다.
	1. paired 세팅은 unpaired 보다 쉽다고 사료됨.
	2. generator와 discriminator 간의 경쟁에서 모델 사이즈는 매우 중요함
	3. Grover가 다른 모델들보다 우수한 성능을 보임
이는 결국 fake news를 구별하는데 있어서, fake news를 생성시킨 generator와 유사한 bias를 가져야만 이를 구분하기 쉽다는 것을 알 수 있습니다.

5.3 Weak supervision: what happens if we don’t have access to Grover-Mega?
위에서 Grover가 같은 적대자에게 있어서 더 좋은 구별자 역할을 수행함을 설명했습니다. 
그렇다면, Grover-mega를 사용하고 p값이 알려지지 않은 적대자를 고려해봅시다. 이 설정에서, 우리는 Grover Base, Large를 이용할 수 있습니다. 
위 결과들은 Grover가 테스트 시에 마주할 똑같은 적대자로부터 생성된 가짜 뉴스의 수가 적당히 많을 때 효과적인 discriminiator로 작용한다고 설명합니다. 그렇다면 이런 가정을 더 낮추면 어떻게 될까요? 따라서, 우리는 Grover-Mega와 알려지지 않은 top-p threshold 0.14를 이용해 만드는 가짜 뉴스를 구분하는 문제를 가정했습니다. 이 문제에서, 우리는 Grover-Mega보다 약한 Grover-Base, Large만 사용할 수 있습니다. 오직 X개의 Grover-Mega의 샘플에만 접근할 수 있고, 5000-x개의 다른 Weaker Model에서의 샘플을 사용합니다. top-p는 [0.9~1.0]의 범위를 갖습니다. *
추가적인 generation을 확인하는 것이 구별자에게 매우 큰 도움이 됨을 알 수 있습니다. *


6. How does a model distinguish between human and machine text?
이 섹션에서 우리는 Grover가 왜 가짜 뉴스를 감지하는데 가장 좋은지를 알아보려 합니다. 그 원인에는 exposure bias와 varian-reduction가 모순 관계에 있기 때문입니다. bias를 낮추는 알고리즘들이 또한 discriminator가 사용할 artifacts들을 만드는 것입니다.

exposure bias
Eq1를 최대화시키는 모델은 오직 사람이 쓴 기사로만 학습이 됩니다. 즉, 모델이 생성한 데이터는 사용하지 않습니다. 이는 exposure bias라는 문제를 일으킵니다.
*exposure bias
body 의 position에 따라 perplexity를 조사했습니다.
<starbody> 다음 바로 첫 토큰을 만드는 것은 높은 perplexity를 초래했습니다. 그러나, 남은 position들은 의문스러운 패턴을 만들어냈습니다. 사람이 쓴 텍스트가 무작위로 샘플링된 텍스트보다 perplexity가 낮고, 이는 sequence 가 길어질수록 차이가 커졌습니다. * 즉 무작위 샘플링은 Grover가 사람의 언어 분포에서 확연히 떨어지게 만드는 것입니다. 반면, 분산값을 제한할 경우 사람보다 더 낮은 perplexity를 만들어냈습니다.

한편, 모델의 분산을 잘라내는 것 또한 artifacts를 남기는데, 이 사례는 (Strobelt and Gehrmann,2019) 를 통해 확인할 수 있습니다. 비슷한 현상이 Nucleus (top-p) sampling 에 나타납니다.* 사람이 쓴 기사의 모든 토큰들이 분포의 top-p에서만 나왔을 확률이 Pn이며, n은 기사의 길이입니다. 이 확률은 n이 증가하며 점점 0으로 갑니다. 그러나, Nucleus Sampled text 는 모든 토큰들이 top-p에서 옵니다. *

구별자를 어떻게 고르느냐에 따라 artifact가 보일지 안보일지 결정됩니다. 만약 구별자가 생성자와 다른 p값을 갖는다면 hard time pinpointing 1-p tail 이는 BERT의 낮은 성능을 설명합니다.

a sweet spot of careful variance reduction
분산값을 줄이지 않는 것 혹은 너무 많이 줄이는 것 또한 문제가 생깁니다. 아마도 이를 조절하는 이상적인 값이 있는 것일까요? 실험 결과는 구별자마다 약 0.92부터 0.98 사이의 이상값을 갖는다고 합니다. 그러나 BERT가 결국 항상 다른 Grover보다 낮은데 이는 top-p 를 낮게 쓴다고 해도 많은 정보를 제공해주는 것은 아니라는 의미입니다.

결국, Grover는 Grover가 만든 가짜 뉴스를 잡는 데 가장 뛰어난데, 이는 tail이 어딨는지 알기 가장 좋기 때문입니다.
