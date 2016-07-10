Includes codes for deep learning architectures aimed at question-answering.

GloVe word embeddings have been used. Run the fetch_glove_data.sh to download the word vectors. Make sure the downloaded text files are directly inside a folder called "data" so that one may run the codes directly. Alternatively, make changes in the code. 

Theano, Gensim and NLTK must be installed for the codes to work.

"Semantic_rel.py" Architecture : ( Relevant Paper :  http://nlp.stanford.edu/pubs/tai-socher-manning-acl2015.pdf )

This model has 2 GRUs ( alternatively LSTMs ) with tied weights for encoding both the sentences. This is followed by a neural network 
to score sentence similarity. There are two types of Neural Networks to choose from for finding this similarity, both of which give approximately the same accuracy. One concatenates both the sentences, their elementwise difference vector and their elementwise product followed by two more layers of neurons. The other performs a linear combination of the elementwise product vector and elementwise difference vector which is again followed by hidden layers to finally produce a score. In both the cases, the loss is calculated using K-L divergence.

"maLSTM.py" : Implementation of Manhattan LSTM as given in - http://www.mit.edu/~jonasm/info/MuellerThyagarajan_AAAI16.pdf

"lex_comp_decomp.py", "lex_rnn", "lex_cnn_attn", "lex_rnn_attn" : ( Relevant Paper : http://arxiv.org/pdf/1602.07019v1.pdf )

Files having rnn in its name use LSTM based mechanism for composition rather than CNN as given in original paper. This modification presents state of the art results on the SICK dataset ( expanded version of SICK by the name superdata.csv ).

Files having "attn" in its name use a Neural Network for scoring similarity as in the previous models. This gives much better and improved results.

"a.txt" is a text file which contains lines written about Kalam copied from Wikipedia and other links without any pre-processing. This is mainly used as the corpus for Question - Answering.

The file "demo.py" provides a demo based on console where one can ask questions about Kalam and get the top 15 candidate answers along with their scores. This final model uses the architecture in the file "lex_rnn_attn.py" trained on "expanded SICK dataset" in combination with BM25 to filter out irrelevant sentences. 

"demo_ui.py" achieves the same as above but provides a User-Interface for easy use. 

The libraries "Gensim" and "NLTK" are used to implement BM25. For implementing the Deep Learning architectures, Theano has been used. 


















