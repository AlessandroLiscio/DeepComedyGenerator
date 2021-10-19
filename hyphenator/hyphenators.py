import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import numpy as np
from collections.abc import Iterable
# from timer import timer
import pickle
import json

VERBOSE = False# True # <-------------------------------- debug 


class hyphenator():
    

    def __init__(self, text_sequences:list=None, hyphenation_mask_sequences:list=None, sep="|", train_params = None, from_pretrained=None):
        """Fits the hyphenator to the desired text BLA BLA BLA TODO 
        text_sequences (list of strings): Sequences of the verses to hyphenate. 
        hyphenation_mask_sequences (list of binary strings): list of strings. 
        For each string of text_sequences, store a string which has the i-th character set to 1 if the i-th token of the corresponding verse is the starting of a new syllable
        """
        self.sep = sep              # separator for the hyphenation. Default "|". E.g. "Nel |mez|zo |del |cam|min |di |no|stra |vi|ta"
        self.vocab_size = None      # dimension of the vocabulary
        # self.mask_token_to_id = {"0":0, "1":1}
        # self.mask_id_to_token = {v:k for k,v in self.mask_token_to_id.items()}
        
        # load tokenizers from pretrained
        if not from_pretrained is None:
            with open(f"{from_pretrained}/tokenizer.pkl", "rb") as f:
                self.tokenizer = pickle.load(f)

            with open(f"{from_pretrained}/mask_tokenizer.pkl", "rb") as f:
                self.target_tokenizer = pickle.load(f)

            with open(f"{from_pretrained}/_confing_.json", "r") as f:
                self.train_params = json.load(f)

            self.max_len = self.train_params["max_len"]
            self.vocab_size = len(self.tokenizer.word_index) + 1

        # define a new model and tokenizer
        else:
            # integrity check
            if text_sequences is None or hyphenation_mask_sequences is None:
                raise Exception("You should either provide text_sequence and hyphenation_mask_sequences, or specify where to load the model from using argument from_pretrained.")
            if len(text_sequences) != len(hyphenation_mask_sequences):
                raise Exception("text_sequences and hyphenation_mask_sequences should have the same length")

            # build tokenizers
            self.tokenizer = Tokenizer(lower=False, char_level=True)
            self.target_tokenizer = Tokenizer(lower=False, char_level=True)
            self.train_params = train_params
            self.use_validation_data = True # <---------------------------------- TODO: che famo?

            # build input sequences
            self.tokenizer.fit_on_texts(text_sequences)
            self.target_tokenizer.fit_on_texts(hyphenation_mask_sequences)
        
            input_sequences = self.tokenizer.texts_to_sequences(text_sequences)
            input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, padding='post')
            
            target_sequences = self.target_tokenizer.texts_to_sequences(hyphenation_mask_sequences)
            target_sequences = tf.keras.preprocessing.sequence.pad_sequences(target_sequences, padding='post')
     
            self.vocab_size = len(self.tokenizer.word_index) + 1
            self.max_len = input_sequences.shape[1]

            # split data
            self.input_train, self.input_test, self.target_train, self.target_test = train_test_split(input_sequences, target_sequences)
            if self.use_validation_data:
                self.input_val, self.input_test, self.target_val, self.target_test = train_test_split(self.input_test, self.target_test)
            

            if VERBOSE:
                print("\nfirst verse\n", input_sequences[0], "\n", self.tokenizer.sequences_to_texts([input_sequences[0]]))
                print("\nhyphenation mask for first verse\n", target_sequences[0])
                print("train samples:      ", len(self.input_train))
                print("validation samples: ", len(self.input_val))
                print("test samples:       ", len(self.input_test))
                print("max sequence lenght:", self.max_len)
                print("vocab size:         ", self.vocab_size)
                print()

        return


    def save(self, to="."):
        if self.model != None:
            # save model
            self.model.save(to)

            # save tokenizer
            path = to + ("/" if to[-1] != "/" else "")
            with open(f"{path}tokenizer.pkl", 'wb') as f:
                pickle.dump(self.tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

            # TODO: remove the target tokenizer
            with open(f"{path}mask_tokenizer.pkl", 'wb') as f:
                pickle.dump(self.target_tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

            # save train parameters
            with open(f"{path}_confing_.json", "w") as f:
                json.dump(self.train_params, f)

        else: raise Exception("Error: model has not been built. ")
        return


    def to_plain_text(self, hyphenated_text: str) -> str:
        """ given a text with hyphenation separators, returns the plain text without them. """
        return hyphenated_text.replace(self.sep, "")


    # def hyphenation_mask_to_sequence(self, hyphenation_mask):
    #     """hyphenation_mask: string"""
    #     return [self.mask_token_to_int[char] for char in hyphenation_mask]


    def apply_syll_mask(self, text:str, mask) -> str:
        """ Use the syllabification mask on the given text to obtain syllabified text """

        def expand_ones(x: tuple):
            """ Given a text char `t` and a mask character `m`, if `m` is 1, prepend syllable separator to `t` """
            t, m = x
            return (self.sep + t) if m == '1' else t

        return ''.join(map(expand_ones, zip(text, mask)))


    def __build_model(self):
        raise NotImplementedError


    def fit(self):
        if VERBOSE:
            print("\n\n\n ===== TRAINING =====\n")
            print("input shape: ", np.array(self.input_train).shape, "\n target shape", np.array(self.target_train).shape)
            print("input: \n", self.input_train[:2])
            print("target: \n", self.target_train[:2])

        # t = timer()
        self.model.fit(self.input_train, 
                        self.target_train,
                        epochs=self.train_params["epochs"], 
                        validation_data=(self.input_val, self.target_val) if self.use_validation_data else None)

        # if VERBOSE:
        #     print("training completed in ", t.partial())
        return 

    def print_model_summary(self):
        self.model.summary()


    


    def hyphenate(self, text, return_mask=False):
        """ calls the model to predict the positions of the syllables in the token list """
        if type(text) == str: text = [text]
        if not isinstance(text, Iterable): raise Exception("text must be Iterable")

        input_text = self.tokenizer.texts_to_sequences(text)
        input_text = keras.preprocessing.sequence.pad_sequences(input_text, padding='post', maxlen=self.max_len)
        hyphenation_masks = (np.argmax(self.model.predict(input_text), axis=-1))
        hyphenation_masks = self.target_tokenizer.sequences_to_texts(hyphenation_masks)
        hyphenation_masks = ["".join(mask).replace(" ", "") for mask in hyphenation_masks]

        hyphenated_text = [self.apply_syll_mask(t, m) for t,m in zip(text, hyphenation_masks)]

        if return_mask:
            if len(input_text) == 1:
                return (hyphenated_text[0], hyphenation_masks[0])
        else:
            if len(input_text) == 1:
                return hyphenated_text[0]
    







class cnnhyphenator(hyphenator):
    """
    Hyphenator implemented using a CNN. 
    """

    def __init__(self, text_sequences:list=None, hyphenation_mask_sequences:list=None, sep:str="|", train_params:dict=None, from_pretrained=None):

        hyphenator.__init__(self, text_sequences, hyphenation_mask_sequences, sep, train_params, from_pretrained)

        # override specific CNN train params if arguments provided <-------- maybe there's a better way to do this??? 
        if train_params != None:
            self.train_params = train_params
        else:
            self.train_params = {
                "max_len": self.max_len,
                "embedding_dim": 256,
                "filters": 250,
                "kernel_size_1": 9,
                "kernel_size_2": 6,
                "hidden_dims": 250,
                "loss": 'sparse_categorical_crossentropy',
                "optimizer": 'adam',
                "metrics": ['accuracy'],
                "epochs": 6
            }        
        self.__build_model(from_pretrained)


    def __build_model(self, from_pretrained=None):
        """ Build a super cool CNN doing all the stuff """
        if from_pretrained is None:
            # integrity check
            if self.vocab_size == None: 
                raise Exception("Error: Unknown vocab_size. ")

            # model initialization
            self.model = tf.keras.Sequential([
                tf.keras.layers.Embedding(self.vocab_size, self.train_params["embedding_dim"], input_length=self.max_len),
                tf.keras.layers.Conv1D(self.train_params["filters"], self.train_params["kernel_size_1"], strides = 1, padding = 'same', activation = 'relu'),
                tf.keras.layers.Conv1D(self.train_params["filters"], self.train_params["kernel_size_2"], strides = 1, padding = 'same', activation = 'relu'),
                tf.keras.layers.Dense(self.train_params["hidden_dims"], activation = 'relu'),
                tf.keras.layers.Dense(3, activation = 'softmax')
                ])
            
            self.model.compile(loss = self.train_params["loss"],
                            optimizer = self.train_params["optimizer"], 
                            metrics = self.train_params["metrics"])
        else:
            self.model = keras.models.load_model(from_pretrained)

        return



    



class rnnhyphenator(hyphenator):
    """
    Hyphenator implemented using a RNN. 
    """

    def __init__(self, text_sequences:list=None, hyphenation_mask_sequences:list=None, sep:str="|", train_params:dict=None, from_pretrained=None):

        hyphenator.__init__(self, text_sequences, hyphenation_mask_sequences, sep, train_params, from_pretrained)

        # override specific CNN train params if arguments provided <-------- maybe there's a better way to do this??? 
        if train_params != None:
            self.train_params = train_params
        else:
            self.train_params = {
                "epochs": 1
            }        
        self.__build_model(from_pretrained)


    def __build_model(self, from_pretrained=None):
        """ Build a super cool RNN doing all the stuff """
        if from_pretrained is None:
            # integrity check
            if self.vocab_size == None: 
                raise Exception("Error: Unknown vocab_size. ")

            # model initialization
            self.model = tf.keras.models.Sequential([
                tf.keras.layers.Embedding(self.vocab_size, 128),
                #tf.keras.layers.GRU(256, activation='relu', return_sequences = True),
                tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, activation='tanh', return_sequences = True)),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(3, activation='softmax'),
            ])
            
            self.model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01), 
                        loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none'), 
                        metrics= ['accuracy'])
        else:
            self.model = keras.models.load_model(from_pretrained)

        return





# class encoderhyphenator(hyphenator):
#     """
#     Hyphenator implemented using a transformer encoder. 
#     """
#     def __init__(self):
#         self.model = None
#         hyphenator.__init__(self)
    
    
    