# Hierarchical Sequence to Sequence Model for Multi-Turn Dialog Generation

A hierarchical sequence to sequence model similar to the hierarchical recurrent encoder-decoder (HRED) in the following paper.
> Iulian Vlad Serban, Alessandro Sordoni, Yoshua Bengio, Aaron C. Courville, and Joelle Pineau. Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models. AAAI 2016.

The model consists of three RNNs. At each time step, the encoder RNN takes one utterance as input and encodes it into a fixed-sized context vector, which is the input to the context RNN, and then the decoder RNN generates the response by decoding the output of the context RNN. Unlike the original paper, we use GRU instead of LSTM, and the context vector serves only as the initial hidden state of the decoder RNN.

## Dataset
We use the [DailyDialog](http://yanran.li/dailydialog.html) dataset, which contains 13,118 multi-turn dialogs in total. In addition, each utterance in each dialog is labeled with one of the Ekman's six emotions plus a neutral one.
> Yanran Li, Hui Su, Xiaoyu Shen, Wenjie Li, Ziqiang Cao, and Shuzi Niu. DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset. IJCNLP 2017.

## How to Run
The model is implemented using TensorFlow. We use Python version 3.

Please download the dataset from the link provided above, and then extract everything from the `ijcnlp_dailydialog.zip` file. Rename the extracted folder to `corpus`. Run `python read.py` to preprocess the text data. To train the model, run `python main.py`. To predict, run `python main.py --predict`, and make sure that you have a space before each punctuation when you type some input text. For various command line arguments, please see the source files.

## TODO
* Multi-layer RNN. Currently only single-layer architecture is implemented.
* Model evaluation. Currently all the data are used for training, and no evaluation method is implemented.
