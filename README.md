# Twitter Sentiment Analysis
 Naïve Bayes model implementation for identifying the sentiment of a tweet.
### Nom: Marc Garrofé Urrutia
### Niu: 1565644
### DATASET: Twitter Sentiment Analysis
### URL: [drive](https://drive.google.com/file/d/1Yf33z87GymmCGdkC5Jnxc8oOoOwLSptz/view?usp=sharing)

## Resum
El dataset conté informació sobre aproximadament 1.5 M de twits classificats segons si aquests són positius o negatius.
Ens faciliten 4 columnes que segueixen aquesta estructura:

Id del missatge | Contingut del missatge | Data del missatge | Classificació del missatge

## Objectius del dataset
Classificar amb Naive Bayes els missatges analitzant les probabilitats de les paraules que formen els missatges.

## Experiments
| Model | Temps fit() | Temps predict() | Dades Dataset | Split Ratio | laplace alpha | Accuracy | Precision | Recall |
| -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Naive Bayes default | 9.22 s | 7.63 s | 1.564.280 | 0.2 | 1 | 0.7515 | 0.6017 | 0.8575 |
| Naive Bayes default | 9.22 s | 7.63 s | 1.564.280 | 0.2 | 2 | 0.7501 | 0.5940 | 0.8641 |
| Naive Bayes Preprocessing | 8.23 s | 6.79 s | 1.564.280 | 0.2 | 2 | 0.7583 | 0.6317 | 0.8457 |
| Naive Bayes Preprocessing | 9.00 s | 7.47 s | 1.564.280 | 0.2 | 1.7 | 0.7609  0.6396 | 0.8434 |

## Demo
Per executar el codi només cal executar des del directori arrel la següent comanda:
```python3 main.py```

## Conclusions
Concluïm que en el nostre model, i generalment en Naive Bayes,
l’eficàcia del model no depèn en tanta mesura de la mida del diccionari, sinó de tenir en el diccionari les paraules claus que classifiquen correctament un twit.

## Llicencia
El projecte s’ha desenvolupat sota llicència MIT.

## Idees per treballar en un futur
Millorar l'apartat de preprocessament. Analitzar els twits per tal de combinar paraules que generen males prediccions, com ara: "Not sad". Aquesta combinació té una alta probabilitat de ser negativa pero realment indica tot lo contrari.
