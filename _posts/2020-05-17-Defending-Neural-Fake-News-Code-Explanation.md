---


layout: post
title: "Defending Against Neural Fake News 코드 살펴보기"
excerpt : "https://github.com/rowanz/grover 분석"
categories:
  - Papers
tags:
  - Papers
  
  
---

![Grover](/asset/translation/tr1.png)
이번 포스트에선 정리했던 논문인 "Defending Neural Fake News" 구현에 대한 부분을 설명하겠습니다.   
[https://github.com/rowanz/grover/](https://rowanzellers.com/grover/)에서 해당 논문에 대한 구현 코드가 게시되어 있습니다. 또 download_model.py 를 이용해 데이터셋과 학습 체크포인트를 다운받을 수 있습니다. 이전에는 Grover-Mega의 체크포인트와 데이터셋을 구글 폼을 작성해야만 배포해주었는데, 최근에 BERT 관련 기술들이 많이 나오면서 배포를 제한하는 것이 무의미하다고 판단한 것 같습니다.  

![Dirs](/asset/code_exp/git_dirs.png)

깃허브를 들어가시면 위와 같이 5개의 디렉토리가 존재합니다.   

discrimination 안에는 classification을 학습하거나 evaluation을 할 수 있는 코드가 게시되어 있고, 기학습되어 있는 체크포인트의 경우 google cloud storage를 이용해 다운로드 받으셔야 합니다.   
generation_examples 안에는 주어진 데이터셋의 정확도를 평가할 수 있는 코드와 generation 체크포인트의 다운로드 방법을 알려주고 있습니다.   
lm은 전체적인 학습 모델을 포함하고 있습니다. tensorflow는 define and run 구조를 따르기 때문에 모델을 기존에 선언해둘 필요가 있는데, 이 정보들이 lm에 담겨있습니다.  
sample은 define된 모델을 실행시키는 코드들이 담겨있습니다.  
이중 Grover Model의 데이터 전처리, 그리고 모델 빌드와 훈련시키는 과정을 살펴보겠습니다.   

<h2>sample 디렉토리 </h2>
april2019_set_mini.jsonl : 데이터 예시 파일  
vocab.bpe : Unicode 처리를 위한 lookup table 용 파일  
encoder.json : corpus 파일  
<h3>encode.py  </h3>
<h4>(1) Encoder 클래스   </h4>
Encoder 클래스는 {domain, date, authors, headline, body} 에 대해 begin 토큰과 end 토큰을 생성합니다.   
유니코드 처리를 위해 byte_encoder, byte_decoder를 만들어 lookup table을 만들어둡니다.   
encode 함수 :   
String을 패턴으로 잘라 토큰으로 만들고, utf-8로 인코딩하고, 이를 bpe lookup table을 이용해 가공합니다.   
![encoding_result](/asset/code_exp/encode_token.png)
encode를 수행할 경우 텍스트가 위 이미지처럼 변환됩니다.  
decode 함수 :   
encoding 됐던 토큰들을 다시 텍스트로 변환시킵니다.   
<h4> (2) tokenize_article_pieces </h4>
위의 인코딩 결과를 보여주는 이미지가 tokenize_article_pieces 함수를 수행시켜 출력해본 결과입니다. 인코더를 불러 데이터를 토큰으로 가공하는 것을 확인할 수 있었습니다.
<h4> (3) tokenize_for_grover_training </h4>
tokenizing을 수행하고, 패딩을 최소화하는 함수입니다. Maximum Sequence가 1024로 정해져 있기 때문에 이를 모든 데이터에 맞추는 Padding 작업을 수행해야 하는 것 같습니다. 그러나 이를 최소화해서 computing time을 최소화 하려고 합니다. 또한, 논문에서 언급한대로 <b>canonical order</b> 로 데이터 순서를 맞춰줍니다. 이 함수는 1개의 기사를 처리하는 코드입니다.     
다만 논문에서는 <b>domain - date - authors - headline - body</b> 라고 명시했는데,   
실제 구현에선    <b>domain - date - authors - title - article - summary </b>라고 순서를 맞췄습니다.   
앞서 설명한 tokenize_article_pieces로 기사 토큰들을 가져옵니다. 
이후 설정한 unconditional_probability에 따라 article 부분을 pop합니다. 논문에서대로, article의 probability는 35%로 정해져있고, 다른 부분들은 10%로 설정되어 있습니다.   
또한 Maximum sequence에 수를 맞추고 이를 초과하는 경우 padding token을 삽입하고 다시 넣는 방식을 수행합니다. 
<h4> (4) extract_generated_target </h4>
![extracted_result](/asset/code_exp/extract_generated_target.png)
위의 결과가 해당 함수를 실행시킨 결과입니다. 즉, 생성된 토큰의 결과들을 다시 text로 변환시키는 작업입니다. Generator의 최종 결과라고 할 수 있습니다.   

<h3>contextual_generate.py </h3>
 contextual_generate.py 파일은 encoder.py 에서 선언한 encoder class와 후술할 GroverConfig에서 모델의 파라미터들을 모두 받아옵니다. 이후 배치사이즈 등 초기 설정을 마치고, tensorflow session을 열어 모델의 generator 체크포인트를 받아옵니다. 이후 데이터를 받아 이를 tokenize하고 인풋에 맞게 처리를 한뒤 session run을 통해 기학습된 체크포인트의 가중치로 행렬곱을 수행합니다. 마지막으로 생성된 결과를 extract_generated_target에 입력할 경우 결과를 확인할 수 있습니다. 실행 결과는 아래와 같습니다.   

<b>input으로 들어온 텍스트</b>   
![cg_input](/asset/code_exp/context_generation_compare1.png)
<b>output으로 나온 결과   </b>
![cg_output](/asset/code_exp/context_generation_compare2.png)
이처럼 유사한 내용이나 문서 내용 자체는 아예 달라졌음을 확인할 수 있습니다.   

<h2>2. lm </h2>
validate.sh , validate.py : 모델의 Perplexity 측정을 위한 코드   

lm 디렉토리는 코드 간의 일부 종속성을 고려한 순서대로 작성하였습니다.   
우선 Grover는 Tensorflow Estimator를 사용하는 것을 알아야 합니다. Tensorflow Estimator는 하이레벨 API를 쉽게 사용하기 위해 사용합니다. Grover는 Custom Estimator를 사용했는데, 이를 위해 input_fn, model_fn, model_fn 내의 train, evaluate, predict 함수를 모두 구현하였습니다.   

<h3>dataloader.py </h3>
<h4> (1)input_fn_builder </h4>
input_fn 이란 함수를 만들기 위한 함수입니다. Estimator 내에서 사용할 input_fn 을 리턴하는 함수입니다.  즉, 내부의 <b>input_fn</b>이 핵심입니다.   
input_fn은 training의 경우엔 데이터셋을 자르고, repeat하고 (epoch 수를 늘림. 명시되어있지만, 코드에선 default 값인 None으로 되어있어 epoch는 1로 사료됩니다. ), 섞어줍니다. 이후 parallel_interleave를 통해 병렬적으로 cycle_length 수 만큼의 nested dataset을 얻는다고 합니다.   
반면 evalute를 위한 input의 경우엔 단순히 input_file을 TFRecordDataset에 바이너리 형식으로 저장합니다.   
이후 batch 사이즈 마다 mapping function을 수행한다고 하는데, 이부분은 TPU를 사용하기 때문에 처리하는 부분입니다. TPU가 32bit만 지원하기 때문에, 64bit integer로 저장된 데이터 값을 모두 32비트로 처리하는 작업을 함께 수행합니다.   
결론적으로, 해당 함수는 tensorflow.data를 리턴하는 input_fn 함수를 빌드하는 함수입니다.   

<h4> (2) classification_input_fn_builder </h4>
위의 input_fn_builder 와 동일한 작업을 수행합니다. 다만 feature에 label_ids, is_real_example이라는 feature가 추가되었습니다. 기존에는 input_ids만을 사용했었습니다.   

<h4> (3) classification_conver_examples_to_features </h4>
Classification을 위해 사용하는 함수입니다. Input을 TFRecord file로 바꾸는 함수입니다. 

<h3>utils.py </h3>
utils.py에서는 뒤에 후술할 modeling.py, train.py 에서 기본적으로 사용하는 간단한 함수들에 대한 정의가 쓰여 있습니다.   
<h4> (1) save_np </h4>
일련의 array를 npy파일로 저장하는 데 사용되는 함수입니다. discrimination 수행 후 predict를 통해 probability를 구하고 이를 저장시에 사용됩니다.   

<h4> (2) assert_rank </h4>
텐서가 원하는 차원의 수와 다를 경우 Error Raise를 수행하는 함수입니다.  

<h4> (3) get_shape_list </h4>
tensor의 shape를 리턴해주는 함수입니다.  

<h4> (4) gelu </h4>
Gaussian Error Linear Unit  
![GELU_img](/asset/code_exp/gelu_image.png)   
수식은 아래와 같습니다.   
![GELU_exp](/asset/code_exp/gelu_expression.png)   
해당 이미지는 모두 GELU 논문에서 가져왔습니다.  

<h4> (5) layer_norm </h4>
layer normalization을 수행하는 함수입니다.  

<h4> (6) dropout</h4>
dropout_prob를 받아 dropout을 수행하는 함수입니다.   

<h4> (7) get_attention_mask</h4>
masking 작업을 수행하는 코드입니다. 아마 현 position보다 뒤에 있는 position에 대해서는 0으로, 앞에 있는 position은 1로 주는 코드로 설정해주는 것 같습니다.   


<h3>optimization_adafactor.py </h3>
해당 파일에서는 adafactor optimizer를 구현해두었습니다.   adafactor에 대한 자세한 설명은
Adafactor: Adaptive Learning Rates with Sublinear Memory Cost 에서 확인하실 수 있습니다.  

<h3>modeling.py </h3>
본 포스트의 핵심이라고 할 수 있는 modeling.py 입니다.   
<h4> (1) GroverConfig 클래스 </h4>
Grover 모델의 Configuration Setting 하는 클래스입니다. 아래는 파라미터의 설명입니다.   
![Grover_config](/asset/code_exp/grover_config.png)
configuration setting 외엔 json 형태로 저장하거나 불러오는 함수를 구현해두었습니다.   

<h4> (2) mask_attention_for_ltr</h4>
학습을 위해 masking 된 부분들의 attention 값을 -10000으로 처리한다는 함수로 사료됩니다.   
masking을 이용해 현재 position보다 뒤에 있는 position에 있는 값을 반영하지 못하게 하기 위함입니다. 아마 GPT는 Unidirectional 이기 때문에 이런 방식으로 처리하는 것 같습니다.      

<h4> (3) attention layer </h4>
Grover의 원조 모델인 GPT2는 Text와 Position의 임베딩된 입력을 Masked Multi Self Attention 기법을 사용해 처리합니다. 코드 내부를 보시면 Query, Key, Value 모두 동일한 x값을 사용하기에 self attention 임을 알 수 있습니다. 이후 Query와 Key를 Matrix Multiplication을 수행하고, 마스킹 작업을 수행한 뒤 Softmax 함수를 취해줍니다. 이후 Value를 다시 Matrix Multiplication을 수행해주는데, Query와 Value가 비슷할수록 높은 값을 가지게 됩니다. 이후 해당 값을 dense layer에 넣어서 multi head attention을 수행합니다.    

<h4> (4) residual_mlp_layer </h4>
![GPT_Architecture](/asset/code_exp/gpt_arc.png)
GPT의 구조를 살펴보면, attention 을 지나고 나면, attention의 결과와 인풋을 더하고, 이를 Layer Normalization에 넣어줍니다. 이후, 이를 Feed Forward Layer에 넣고, 다시 한번 Input과 결과를 더해준 뒤 마지막으로 Layer Normalization을 재수행하는 구조를 가집니다. 이러한 과정을 수행하는 것이 residual_mlp_layer입니다.   

<h4> (5) embed </h4>
embedding 수행되는 함수입니다. Grover Model의 인풋에 대해서 실행합니다. Embedding을 한다는 것은 Text와 Poistion에 대해 값을 더해주는 과정을 뜻합니다. 문서 내의 단어의 Position 또한 영향을 주기 때문입니다. 이후 layer_normalization을 수행하고 리턴합니다.   

<h4> (6) GroverModel 클래스 </h4>
Grover는 GPT2와 동일한 구조를 따르는데, GPT2는 Decoder 구조로만 이루어진 Transformer 라고 할 수 있습니다. 
먼저 embed 함수를 실행시켜 input에 대해서 positional vector와 text의 vector를 연산해줍니다. 이후 get_attention_mask를 실행시켜 마스킹된 벡터를 받습니다. 이후 이를 attention_layer에 넣어주고 연산한 뒤, 계산 결과를 처리해주는 residual_mlp_layer에 넣어 하나의 transformer architecture를 통과시킵니다. 이후, caches의 사이즈만큼 해당 작업이 반복됩니다. 아마도 연속적으로 transformer 구조를 쌓는 것 같습니다.  
![multiple layers](/asset/code_exp/attention_multi.png)
위처럼 실제로 train.py 를 실행시켜보면 chache 사이즈가 12일때 layer 11까지 쌓이는 것을 확인할 수 있습니다. 처음에는 저도 이 부분이 MultiHead Attention을 구현하려고 만든 부분이라고 생각했는데, 그림에서 같은 layer에 여러개의 K, Q, V가 존재하는 것 보면 attention layer에서 multi head 부분을 처리하는게 맞고, 작업 반복은 트랜스포머를 여러 개 쌓는게 맞는 것 같습니다.   
이외에도 loss를 계산하기 위한 함수도 따로 구현해두었습니다.   

<h4> (7) top_p_sampling & top_k_sampling </h4>
top_p 와 top_k 샘플링 방식을 구현해둔 함수입니다. 이전 포스트에서 설명했듯이, top_p는 threshold p가 만족될 때까지 단어의 생성확률을 누적합해주는 것이고, top_k는 k개의 selection을 만족할 때까지, 유력한 후보 k개의 단어를 선택하는 방식입니다.   

<h4> (8) model_fn_builder</h4>
이전에 input_fn_builder 처럼 Estimator로 전달해주기 위해 model_fn 함수를 빌드하는 함수입니다.   
구현한 GroverModel를 임포트하고 Loss function과 Optimizer를 정의합니다. 이후, tf.estimator.Modekeys 의 값에 따라 따로 처리하는데, Train,Eval,Predict 값을 가지므로 이에 대한 각각의 파트를 구현하였습니다. 다만 EstimatorSpec에 대해서 정의만 하고, 해당 Spec를 리턴해줍니다. (실행은 다른 함수에서 하니까)   

<h4> (9) classification_model_fn_builder</h4>
Classification을 위한 Estimator 함수 빌드용 함수입니다.   
이전 포스트에서 Classification을 하기 위해선, article의 마지막 토큰을 (cls) 로 놓고 이 hidden state를 비교함으로써 알 수 있다고 언급했습니다. 이에 관한 코드는 아래와 같습니다.   
<script src="https://gist.github.com/ptaejoon/b2a208bc8018bbb2410cb9c14f92060f.js"></script>   

<h4> (10) sample & sample step</h4>
모델의 결과로부터 top-p 혹은 top-k 방법을 사용해 단어(현재는 아직 토큰 번호)를 선정, 결과를 만드는 함수입니다.   

<h3>train.py </h3>
train.py 는 모델을 학습시키는 코드입니다. 그동안 설명한 모든 부분을 이용해 세팅부터 Estimator의 train 함수를 부르기까지의 과정을 포함합니다.   
즉, GroverConfig 클래스를 이용해 모델 환경설정을 하고, 결과물을 도출할 Output 디렉토리를 만들고, Estimator를 위해 model_fn을 빌드, TPUEstimator를 설정 후 train을 실행시킵니다.  
지금까지의 lm 디렉토리의 모든 요소를 담고 있습니다.   
<script src="https://gist.github.com/ptaejoon/f7ffb4bd236d13cc37f44b97e30ab071.js"></script> 




<h2> Top P sampling 과 Top K sampling 비교해보기 </h2>
Github에서 제공하는 기본적인 예시를 따라해봤는데, Top K sampling을 쉽게 비교해볼 수 있어서 한번 실행시켜보았습니다.   
![toppk](/asset/code_exp/topp_topk.png)   
해당 예시의 헤드라인은 멀웨어 연구자 Marcus 가 유죄 선고를 받았다는 내용인데요. Sampling Generation을 위해 위의 이미지에는 없지만 Generation을 할때 기사의 내용도 함께 입력받았습니다. 이를 바탕으로 Text Generation을 해보았는데, 언뜻 보기에도 Top-p sampling이 훨씬 우수한 성능을 내는 것을 확인할 수 있습니다.   
또한 Top-k 방식이 너무 결과가 안좋아서 GPT2 의 코드도 살펴보았는데, GPT2 역시 기본적으로 Top-p sampling을 하는 것을 확인했습니다.   
<script src="https://gist.github.com/ptaejoon/cb92841ece080d552fc1be01a8c97d36.js"></script>

<h2> Top P 를 낮출 경우? </h2>

![low_p](/asset/code_exp/lowp.png)   

Top-P를 0.5로 낮춘 경우에도, 생성되는 텍스트가 어색함은 별로 없는 것 같습니다. 그러나 내용이 완전히 바뀌었습니다. 입력된 텍스트의 경우 악성코드를 퍼뜨린 보안연구자의 유죄선고가 주된 내용인데, p 가 0.9 이상이었을 때는 이를 잘 반영하여 출력하였습니다. 반면에, p가 0.5인 상태로 generation을 할 경우 FBI에 위증을 한 죄로 선고되었다는 내용으로 뒤바뀌었습니다. 
