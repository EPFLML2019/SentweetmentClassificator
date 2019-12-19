# Machine Learning - Project 2: Twitter Sentiment Classification
## How to run our project
To run our project the following libraries are required:
- `tensorflow 2.0.0` or higher
- `gensim`
- `numpy`
- `pandas`
- `ntpath`
- `nltk`
- `pickle`
- `sklearn`

You also need the GloVe dataset from https://nlp.stanford.edu/projects/glove/. Please download the code `GloVe-1.2.zip` extract the archive in the `script` folder and run the command `make` inside the `GloVe-1.2` folder.

By running the file `run.py` you should see the tweets being imported, pre-processed then used to train our model. Finally the prediction should come out in the file `predictions_out.csv`.

## Content of the submission
The folder `scripts`contains:
- `run.py` which provides our final submission
- `test_models.py` which trains a sep-CNN, BLSTM and SVM model and evaluates their performance over on the provided dataset.
- A `preprocessing` folder which contains:
	- `bigrams.py` which computes 2-grams of the embeddings if necessary
	- `embeddings.py` which generates the tweet embeddings
	- `tokenizer.py` which computes the tokens of each tweet
	- `tools.py` which was provided with the project and loads the data, creates the submission and generates the correct pr
ediction format given the output the a model.
- A `models` folder which contains:
	- `bert.py` which sounded like a promising model that we didn't have enough computing power (or time) to test. We include it here for completeness.
	- `blstm.py` which creates a Tensorflow Bidirectional Long Short-Term Memory Recurrent Neural Network model.
	- `sepCNN.py` which creates a Tensorflow Depth-wise Separable Convolutional Neural Network model.

