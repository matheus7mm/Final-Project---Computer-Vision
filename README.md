# Final Project - Computer Vision

## Como utilizar

### 1. Criação de Pastas

Em primeiro lugar, é necessário criar as seguintes pastas:
* models
* graphs
* case 1, case 2, case 3, [...], case 16

### 2. Geração de novos datasets utilizando data augmentation

O arquivo *augmentation.py* é responsável por criar novos datasets para cada um dos casos estudados no trabalho.

Sua execução é feita da seguinte forma:

$ python augmentation.py --dataset "dataset" --augmentation "case x" --case x

Onde x é o número do caso: 1, 2, 3, [...], 16.

Em cada execução, são criadas novas imagens utilizando os métodos de data augmentation de cada caso e são salvos nas suas pastas respectivas. É preciso executar o comando para cada um dos casos.

### 2. Treinamento da rede neural

O arquivo *train.py* é responsável pelo treinamento de cada dataset. Em cada pasta dos datasets, existem subpastas que representam as classas das imagens. Os testes foram feitos utilizando três classes: Coca-cola, Lacoste e Starbucks.

Para treinar cada dataset basta executar o comando:

$ python train.py --dataset "case x" --model models/model_casex.model --labelbin models/lb_casex.pickle --plot "graphs/casex.png"

O comando pega a pasta referenciada em --dataset, que será treinada, e após o treinamento da rede, o seu modelo e suas labels serão salvas nos arquivos de --model e --labelbin. --plot é responsável por salvar o arquivo do gráfico de loss e accuracy do treinamento.

É preciso executá-lo para cada um dos casos, x é o número do caso: 1, 2, 3, [...], 16. E também para o dataset original, que não foi alterado utilizando métodos de data augmentation. Para este, caso o comando seria por exemplo:

$ python train.py --dataset "dataset" --model models/model.model --labelbin models/lb.pickle --plot "graphs/plot.png"

Após as execuções dos comandos, basta conferir os modelos das redes na pasta *models* e os gráficos em *graphs*

### 3. Resultados

O arquivo *result.py* é responsável por realizar o teste de cada modelo de rede neural criado e treinado. Para executá-lo, basta utilizar o comando:

$ python result.py --model models/model_casex.model --labelbin models/lb_casex.pickle --examples test

Em --model e --labelbin, os modelos treinados são referenciados. x é o número de cada caso: 1, 2, 3, [...], 16. Em --examples é passado a pasta que contém imagens para os testes. No presente estudo, foi criado uma pasta *test* contendo 120 imagens, sendo 60 para cada classe. 

Vale ressaltar que as imagens para testes são bem diferentes das imagens utilizadas para o treino. Elas são imagens da utilização das logomarcas, o que dificulta o resultado, pois no treinamento foram utilizados apenas imagens das logomarcas. Por exemplo, em cada classe foram colocadas imagens no dataset apenas da logomarca. Já nas imagens para o teste foram colocadas imagens da utilização dessa logomarca, como ela em uma camiseta, boné ou sapato.
