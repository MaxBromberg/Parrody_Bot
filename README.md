### Parody Parroting: 
Basic, unoptimized natural language processor which may take in a training text and output predictions based on this text 
and a few seed words. Adapted from a [elon musk tweet synthesizer][1]<sup id="a1">[1](#f1)</sup>, the present implementation
creates the fixed model structure 

```
Model: "sequential"
________________________________________________
Layer (type)                 Output Shape       
================================================
embedding (Embedding)        (None, *, 80)     
________________________________________________
lstm (LSTM)                  (None, *, 100)    
________________________________________________
lstm_1 (LSTM)                (None, 50)         
________________________________________________
dropout (Dropout)            (None, 50)         
________________________________________________
dense (Dense)                (None, 92)         
________________________________________________
dense_1 (Dense)              (None, 1852)       
================================================
```

where *, `None` are replaced by the maximum number of words in a single line, total words in the training text respectively.
Presently, `law_giver.py` and `solomon_generator.py` are trained on Leviticus and Proverbs, respectively<sup id="a1">[1](#f1)</sup> 
and serve as a basic test. Ideally, I'll get around to both optimizing and gui-fying this basic NLP...



<!---- References: ---->
[1]: https://www.analyticsvidhya.com/blog/2020/10/elon-musk-ai-text-generator-with-lstms-in-tensorflow-2/
[2]: https://www.kaggle.com/phyred23/bibleverses


<!---- Footnotes: ---->
<b id="f1">1:</b> Thus the _chirp_ and other tweet-related terminology.

<b id="f1">2:</b> [King James Version][2]
