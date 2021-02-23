## Parody Parroting: 
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

<br/>

### Humorous Examples:
Note that punctuation has been omitted from the training data. The format for predictions is:
**seed text** predicted text  

##### 300 Epochs of training on [Proverbs][3]:

**my son forget not** my law but let thine heart keep my commandments and live and my law as the apple of thine eye let thine heart keep thine heart and bow thine heart of knowledge and apply thine heart to understanding thou ways 

**my daughter forget not** thy father's ways and live and be of a humble man and a dross against the seven and he that taketh a wind to the good left his father's way but he that keepeth company with harlots spendeth his substance

**love thy neighbor and** if thou hast thought evil lay thine hand upon thy mouth lest i be filled with mischief but they that keep my ways are consumed and she that soweth me i have me retain my counsel through my love naughtiness

<br/>

##### 300 Epochs of training on [Leviticus][4]:

**if a man and his wife** shall be holy from him or committed thing that creepeth upon the earth that it is did the food of the meat offering shalt thou

**thou shall** stone it with stones and the lord hath spoken unto them unto the lord in the wilderness of sinai two years unto the lord unto

<br/>

<!---- References: ---->
[1]: https://www.analyticsvidhya.com/blog/2020/10/elon-musk-ai-text-generator-with-lstms-in-tensorflow-2/
[2]: https://www.kaggle.com/phyred23/bibleverses
[3]: data/Proverbs.csv
[4]: data/Leviticus.csv

<!---- Footnotes: ---->
<b id="f1">1:</b> Thus the _chirp_ and other tweet-related terminology.

<b id="f1">2:</b> [King James Version][2]
