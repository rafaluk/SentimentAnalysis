from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
import argparse
import os
from argparse import RawTextHelpFormatter


def load_reviews(file_name):
    """Loads data from CSV file to pandas DataFrame.

    # Arguments:
        file_name: name or path with name to CSV file.
    """

    csv_content = pd.read_csv(file_name, sep='\t', index_col=0).sort_index()
    return csv_content


def convert_input_to_x_and_y(input_x, input_y):
    """Converts raw texts from CSV into desired format
    for further analysis.

    # Arguments:
        input_x: column with words.
        input_y: column with labels.
    """

    # Get rid of apostrophes, comas and double spaces
    x = input_x.apply(lambda x: x.replace("\'", "").replace(",", "").replace("   ", "  ").replace("  ", " ")[1:-1])
    x = list(x)

    # Convert sentiments to numpy array
    y = input_y.values

    # Transform categorical variable into vectors:
    #  1 := [1 0 0]
    #  0 := [0 1 0]
    # -1 := [0 0 1]
    le = LabelEncoder()
    y = le.fit_transform(y)
    y = to_categorical(y)

    return x, y


def split_x_y(padded_x, y):
    """Splits x and y sets using sci-kit into:
        - train sets,
        - test sets,
        - validation sets.
    # Arguments:
        padded_x: column with words encoded to integers in form
            of vectors with equal lengths.
        y: column with output values.
    """

    # Test size 20% as a rule of thumb
    x_train, x_test, y_train, y_test = train_test_split(padded_x, y, test_size=0.2)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

    return x_train, x_test, x_val, y_train, y_test, y_val


def create_set_with_all_words_from_input(tokenizer):
    """Creates a set with all unique words from input data.
    It is necessary to effectively assign embeddings to
    these words.

    # Arguments:
        tokenizer: a tokenizer fit previously on input.
    """
    word_index_all = set()
    for key, value in tokenizer.word_index.items():
        word_index_all.add(key)
    return word_index_all


def load_embeddings_from_file(file_name, all_input_words):
    """Loads pre-trained embeddings from specified file.

    # Arguments:
        file_name: name or path with name to file with embeddings.
        all_input_words: set with all words from input data.
    """

    embeddings_dict = dict()

    # Open the file with specified encoding
    with open(file_name, encoding='utf8') as file:

        # Assumption: first row is a header, so skip it
        next(file)

        # Iterate the file over its lines
        for line in file:
            content = line.split()
            word_in_line = content[0]

            # Use only embeddings to words from input
            if word_in_line not in all_input_words:
                continue

            # Do not take the word itself, only coefficients
            coefs = np.array(content[1:])

            # Convert string type to float
            coefs = coefs.astype(np.float)
            embeddings_dict[word_in_line] = coefs

    # Get length of any embedding vector
    any_emb_vector = next(iter(embeddings_dict.values()))
    emb_vec_len = len(any_emb_vector)

    return embeddings_dict, emb_vec_len


def create_matrix_from_embeddings(tokenizer, embeddings,
                                  matrix_size, emb_vec_len):
    """Returns a matrix with embeddings assigned to words from
    input data.

    Firstly the function creates a matrix with zeros only. This matrix
    has as many zeros for all words as long is each embedding
    vector. Then it assigns embeddings to specific vectors. Their
    positions are taken from tokenizer, which had been previously fit.

    # Arguments:
        tokenizer: a tokenizer fit previously on input.
        embeddings: dictionary with all embeddings
        matrix_size: number of words from input data
        emb_vec_len: length of embedding vector
    """

    embedding_matrix = np.zeros((matrix_size, emb_vec_len))

    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


def create_model(model, embedding_matrix, rnn_dim, emb_size, max_emb_input_len, emb_vec_len):
    """Adds layers to the model.

    # Arguments:
        model: model to add layers onto.
        rnn_dim: dimensionality of LSTM cell.
        emb_size: number of words from input data
        max_emb_input_len: maximum length of input vector
        emb_vec_len: length of embedding vector

    # Layers:
        Embedding: non-trainable layer with embedding vectors and weights
        LSTM: Long-Short Term Memory layer as a main deep learning layer
        Dense: regular densely-connected NN layer.
    """

    # Embedding layer
    model.add(Embedding(emb_size, output_dim=emb_vec_len,
                        weights=[embedding_matrix],
                        input_length=max_emb_input_len,
                        trainable=False))

    # LSTM layer with dropout and recurrent dropout defined
    model.add(LSTM(rnn_dim, dropout=0.5,
                   recurrent_dropout=0.5))

    # Dense layer with Softmax activation function
    model.add(Dense(3, activation='softmax'))

    # Use Adam optimizer and Categorical Crossentropy for compilation
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Print summary table
    model.summary()


def fit_model(model, X_train, X_val, y_train, y_val,
              epochs, batch_size, verbose_level):
    """Start training the model with appropriate parameters.

    # Arguments:
        model: model to be trained.
        X_train, X_val, y_train, y_val: training and validation sets.
        epochs: number of training epochs (default: 10).
        batch_size: size of a single batch (default: 32).
        verbose_level: Verbosity mode. 0 = silent,
            1 = progress bar, 2 = one line per epoch.
    """

    # Fit the model with specified arguments.
    # Use EarlyStopping if accuracy improvement is not sufficient
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        callbacks=[EarlyStopping(monitor='acc',
                                                 min_delta=0.0001)],
                        shuffle=True,
                        validation_data=(X_val, y_val),
                        verbose=verbose_level)


def evaluate_model(model, X_val, y_val, verbose_level):
    """Model post-assessment.

    # Arguments:
        model: trained model to be evaluated.
        X_val, y_val: sets for validation.
        verbose_level: Verbosity mode. 0 = silent,
            1 = progress bar, 2 = one line per epoch.
    """

    score = model.evaluate(X_val, y_val, verbose=verbose_level)
    print("Test Loss: %.2f%%" % (score[0] * 100))
    print("Test Accuracy: %.2f%%" % (score[1] * 100))


def main(data_path, embedding_path, rnn_dim, save_dir,
         epochs, batch_size, verbose_level):
    """Main script function, which can be called directly
    from the command line. It performs all tasks necessary
    to run the script accordingly to the specified points.

    # Arguments:
        data_path: path to the file with data in the format
            identical with example dataset
        embedding_path: path to pre-trained embeddings
        rnn_dim: number of hidden units in LSTM layer
        save_dir: path to directory for trained model saving
        epochs: number of training epochs (default: 10).
        batch_size: size of a single batch (default: 32).
        verbose_level: Verbosity mode. 0 = silent,
            1 = progress bar, 2 = one line per epoch.
    """

    print("Parameters:")
    print("\tData path:", data_path)
    print("\tEmbedding parh:", embedding_path)
    print("\tRNN dimensions:", rnn_dim)
    print("\tSaving directory:", save_dir)
    print("\tEpochs:", epochs)
    print("\tBatch size:", batch_size)
    print("\tVerbose level:", verbose_level)

    # Load CSV file with reviews
    print("Loading CSV file...")
    df = load_reviews(data_path)
    print("CSV file loaded.")

    # Process X and y in terms of format and specific chars usage
    X, y = convert_input_to_x_and_y(df['tokens'], df['sentiment'])

    # Define tokenizer, fit it on input data
    # and convert this data to sequences
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)
    encoded_X = tokenizer.texts_to_sequences(X)
    print("All words tokenized.")

    # Define vocabulary size, i.e. number of all words.
    # '+ 1', because '0' index is reserved and cannot be used
    vocab_size = len(tokenizer.word_index) + 1
    print('Found %s unique tokens.' % len(tokenizer.word_index))

    # Get max length of encoded_X vector for calculating padding space
    max_length = len(max(encoded_X, key=len))
    padded_x = pad_sequences(encoded_X, maxlen=max_length, padding='post')

    # Create set with all unique words from input data
    word_index_all = create_set_with_all_words_from_input(tokenizer)

    # Split input data into six sets
    x_train, x_test, x_val, y_train, y_test, y_val = split_x_y(padded_x, y)
    print("Words split into sets:")
    print("\tx_train shape: " + str(x_train.shape))
    print("\tx_test shape: " + str(x_test.shape))
    print("\tx_val shape: " + str(x_val.shape))
    print("\ty_train shape: " + str(y_train.shape))
    print("\ty_test shape: " + str(y_test.shape))
    print("\ty_val shape: " + str(y_val.shape))

    print("Loading embeddings file...")
    embeddings, emb_vec_len = load_embeddings_from_file(embedding_path, word_index_all)
    print("Embeddings file loaded and processed.")
    print('Found %s word vectors.' % len(embeddings))

    # Create embedding matrix
    embedding_matrix = create_matrix_from_embeddings(tokenizer, embeddings,
                                                     matrix_size=vocab_size,
                                                     emb_vec_len=emb_vec_len)

    # Define and process the model
    sequential = Sequential()
    create_model(sequential, embedding_matrix,
                 rnn_dim=rnn_dim, emb_size=vocab_size,
                 max_emb_input_len=max_length,
                 emb_vec_len=emb_vec_len)

    # Model fitting
    print("Model training...")
    fit_model(sequential, x_train, x_val, y_train, y_val,
              epochs=epochs, batch_size=batch_size,
              verbose_level=verbose_level)
    print("Model trained.")

    # Model evaluation
    print("Evaluating the model...")
    evaluate_model(sequential, x_val, y_val,
                   verbose_level=verbose_level)

    # Model saving
    save_path = save_dir + '\sentiment_model.h5'
    sequential.save(save_path)
    print("Model saved to: ", save_path)


def dir_path(path):
    """Function for proper directory definition. Used only for argparse."""
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path.")


def create_parser():
    """Function for command line support."""
    parser = argparse.ArgumentParser(description="Script for Deep Learning task, satisfying following:\n"
                                                 "Your task is to build python script for text "
                                                 "classifier training for Polish language using "
                                                 "simple recurrent neural network with LSTM cell.\n\n"
                                                 "Deep learning implemented with Keras API.\n\n",
                                     epilog="Script written by Rafal Klat.\n"
                                            "\tmail: rafal.klat@gmail.com\n"
                                            "\tphone: +48 664 495 049",
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument('-d', '--data_path',
                        help='Path to the file with input data.',
                        required=True, type=argparse.FileType('r'))
    parser.add_argument('-e', '--embedding_path',
                        help='Path to the file with pre-trained embeddings.',
                        required=True, type=argparse.FileType('r'))
    parser.add_argument('-r', '--rnn_dim',
                        help='Number of hidden units in LSTM layer.',
                        required=True, type=int)
    parser.add_argument('-s', '--save_dir',
                        help='Path to a saving of trained model.',
                        required=True, type=dir_path)
    parser.add_argument('-E', '--epochs',
                        help='Number of training epochs (optional).\n(default: 10)',
                        required=False, type=int, default=10)
    parser.add_argument('-b', '--batch_size',
                        help='Size of a single batch (optional).\n(default: 32)',
                        required=False, type=int, default=32)
    parser.add_argument('-v', '--verbose_level',
                        help='Verbosity mode.\n0 = silent, 1 = progress bar, 2 = one line '
                             'per epoch (optional).\n(default: 0)',
                        required=False,
                        type=int, choices=[0, 1, 2], default=0)
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = vars(parser.parse_args())

    data_path = args["data_path"].name
    embedding_path = args["embedding_path"].name
    rnn_dim = args["rnn_dim"]
    save_dir = args["save_dir"]
    epochs = args['epochs']
    batch_size = args['batch_size']
    verbose_level = args['verbose_level']

    print("-" * 65)
    print("Starting SENTIMENTAL PYTHON SCRIPT WITH Keras LSTM by Rafal Klat.\n")
    print("-" * 65)

    main(data_path, embedding_path, rnn_dim, save_dir, epochs, batch_size, verbose_level)
