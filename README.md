# Checkpoint 2 – Redes Neurais com Keras

## Exercício 1 – Classificação (Wine Dataset)
- Rede neural com 2 camadas ocultas de 32 neurônios (ReLU) e saída Softmax.
- Comparação com RandomForest.
- Métrica: Acurácia.

| Modelo              | Acurácia |
| ------------------- | -------- |
| Rede Neural (Keras) | **1.00** |
| RandomForest        | **1.00** |

* Os dois modelos alcançaram **acurácia perfeita** no conjunto de teste.
* Isso sugere que o problema é relativamente simples para ambos: o dataset tem poucas instâncias e as classes são bem separáveis.

**Conclusão:** Nenhum modelo supera o outro; ambos atingiram o desempenho máximo possível no teste.

## Exercício 2 – Regressão (California Housing)
- Rede neural com camadas 64, 32 e 16 (ReLU) e saída Linear.
- Comparação com LinearRegression.
- Métrica: RMSE.


| Modelo              | RMSE     |
| ------------------- | -------- |
| Rede Neural (Keras) | **0.51** |
| LinearRegression    | 0.75     |

* A rede neural teve **erro médio quadrático raiz (RMSE) menor**, ou seja, suas previsões ficaram mais próximas dos valores reais.
* Isso indica que o modelo não-linear conseguiu capturar padrões mais complexos do que a regressão linear simples.

**Conclusão:** Para este problema, a rede neural apresentou melhor desempenho, oferecendo previsões mais precisas.
