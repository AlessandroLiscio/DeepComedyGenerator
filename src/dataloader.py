import pickle
import os
import tensorflow as tf
from src.tokensprocessing import *

class DataLoader():

    def __init__(self,
                in_path:str = 'data/tokenized/',
                comedy_name:str = 'comedy',
                tokenization:str= 'base',
                repetitions_production:int = 0,
                repetitions_comedy:int = 0,
                from_pickle:str = None,
                verbose:str = True):

        self.comedy_name = comedy_name
        self.tokenization = tokenization
        self.datasets = {'production': None, 'comedy': None}

        if from_pickle:
            self.load(from_pickle, verbose)
        else:

            # Initialize data
            self.vocab = []
            self.files_dict = {}
            
            # initialize datasets parameters
            self.repetitions_production = repetitions_production
            self.repetitions_comedy = repetitions_comedy
            self.train_order = []
            self.original_length_production = 0
            self.original_length_comedy = 0
            self.tercet_max_len = 0

            # initialize separator based on tokenization type
            if tokenization == 'base':      self.separator = ' '
            elif tokenization == 'spaces':  self.separator = '|'
            else:
                print(f"ERROR: incorrect tokenization parameter '{tokenization}'")
                return

            # initialize dataloader's vocabulary, mappings and datasets
            self._init_train_order()
            self._read_files(in_path)
            self._init_vocab_and_mappings()
            self._init_datasets()

            if verbose: print(self)

    def __str__(self):
        return "\n".join((
            "",
            ">> DATALOADER:",
            f"> PARAMETERS",
            f" - comedy_name: {self.comedy_name}",
            f" - tokenization: {self.tokenization}",
            f" - separator: {self.separator}",
            f" - repetitions_production: {self.repetitions_production}",
            f" - repetitions_comedy: {self.repetitions_comedy}",
            f" - original_length_production: {self.original_length_production}",
            f" - original_length_comedy: {self.original_length_comedy}",
            f" - tercet_max_len: {self.tercet_max_len}",
            "> TRAINING ORDER",
            "\n".join(([f" {i+1}- {self._get_clean_filename(filename)}" for i, filename in enumerate(self.train_order)])),
            "> VOCABULARY",
            "\n".join(([f" - {key}: {attr}" for key, attr in self.vocab_info.items()])),
            ""
        ))

    ############################################################################
    #####################          VOCABULARY          #########################
    ############################################################################

    def _init_train_order(self):

        '''initializes the generator training order, based on the
        training epochs for the production and comdedy datasets'''

        if self.repetitions_production > 0:
            for filename in ['convivio','vita', 'detto','fiore']:
                self.train_order.append(self._get_tokenized_filename(filename))
        if self.repetitions_comedy > 0:
            self.train_order.append(self._get_tokenized_filename(self.comedy_name))

    def _read_files(self, directory:str):

        '''reads all text files in 'directory' and initializes the files
        dictionary, with files names as keys and texts as values'''

        files_list = []
        files_names = []

        for i, filename in enumerate(self.train_order):

            # Read file
            path_to_file = os.path.join(directory, filename)
            text = open(path_to_file, 'r').read().lower()

            # generate list from text by splitting lines
            text_list = text.splitlines()
            files_list.append(text_list)
            files_names.append(filename)

        self.files_dict = {files_names[i]:files_list[i] for i in range(len(files_names))}


    def _init_vocab_and_mappings(self):

        '''initialize vocabularies and mappings'''

        #############################
        # final vocabulary order:   #
        # - special tokens          #
        # - punctuation             #
        # - non-ending syllables    #
        # - ending syllables        #
        #############################

        syllables = sorted(
            set().union(*[self._verses_to_syllables_set(self.files_dict[file_name]) for file_name in self.train_order]),
            key=len)

        # initialize groups
        special_tokens = []
        punctuation = []
        non_ending_sylls = []
        ending_sylls = []

        for token in syllables:
            # NON-SYLLABLES
            if '<' in token:
                # print("special:",token)
                special_tokens.append(token)
            elif len(token) == 1 and not token.isalpha():
                # print("punctuation:",token)
                punctuation.append(token)
            # SYLLABLES
            else:
                if not token == '' and not token[-1] == " ":
                    non_ending_sylls.append(token)
                else:
                    ending_sylls.append(token)

        # sort groups
        special_tokens = sorted(special_tokens, key=len)
        punctuation = sorted(punctuation, key=ord)
        non_ending_sylls = sorted(non_ending_sylls, key=len)
        ending_sylls = sorted(ending_sylls, key=len)

        # create the tokens vocabulary following the order
        for group in [special_tokens, punctuation, non_ending_sylls, ending_sylls]:
            self.vocab.extend(group)

        # insert the empty string at poistion 0
        if '' in self.vocab: self.vocab.remove('')
        self.vocab.insert(0, '')

        # store vocabulary information
        self.vocab_info = {
            'size' : len(self.vocab),
            'special tokens' : len(special_tokens),
            'punctuation' : len(punctuation),
            'non-ending syllables' : len(non_ending_sylls),
            'ending syllables' : len(ending_sylls)
        }

        # Creating a mapping from unique characters to indices
        self.str2idx = {u:i for i, u in enumerate(self.vocab)}
        self.idx2str = np.array(self.vocab)

    # Returns set of syllales from input list of verses
    def _verses_to_syllables_set(self, verses_list, verbose:bool=False):
    
        syllables = split_tokens(verses_list, self.separator)
        syllables = flatten(syllables)
        syllables = sorted(set(syllables), key=len)

        if verbose:
            print(syllables)
            print("syllables set: ", len(syllables))

        return syllables


    ############################################################################
    #######################         DATASETS          ##########################
    ############################################################################

    def _init_datasets(self):

        '''creates the dataset to be fed to the generator'''

        for key, dataset in self.datasets.items():

            ## Production dataset
            if key == "production" and self.repetitions_production > 0:

                # Append all productions texts
                dataset = []
                for filename in self.train_order:
                    if not filename == self._get_tokenized_filename(self.comedy_name):
                        dataset.append(self.files_dict[filename])

                # Split input target for Dante' Production dataset
                dataset, self.original_length_production, _ = self._split_input_target(
                    dataset_name = key,
                    dataset = dataset, 
                    inp_len = 3, tar_len = 3,
                    repetitions = self.repetitions_production)

            ## Comedy dataset
            elif key == "comedy" and self.repetitions_comedy > 0:

                dataset = self.files_dict[self._get_tokenized_filename(self.comedy_name)]

                # Split input target for Divine Comedy dataset
                dataset, self.original_length_comedy, self.tercet_max_len = self._split_input_target(
                    dataset_name = key,
                    dataset = dataset,
                    inp_len = 3, tar_len = 4,
                    repetitions = self.repetitions_comedy)

            self.datasets[key] = dataset


    def _split_input_target(self, dataset_name:str, dataset, 
                            inp_len:int=3, tar_len:int=4, skip:int=1, 
                            batch_size:int=64, repetitions:int=100):

        '''splits dataset in input-target couples'''

        inputs = []
        targets = []

        # Concatenate the text lists in production
        if dataset_name == 'production':
            dataset = flatten(dataset)
        
        # Prepare data for model (list of integers)
        dataset = split_tokens(dataset, self.separator)
        dataset = encode_tokens(dataset, self.str2idx)
        
        # Split input-target
        for i in range(0, len(dataset)-tar_len, skip):
            inputs.append(flatten(dataset[i:i+inp_len]))
            targets.append(flatten(dataset[i:i+tar_len]))
            
        # Create repeated, shuffled and prefetched dataset
        real_size, dataset = self._create_dataset(inputs, targets, batch_size, repetitions)
        
        # Real dataset size (not repeated)
        real_size_batched = int(real_size/batch_size)

        # Max length of verses
        max_len = max(len(x) for x in inputs)

        return dataset, real_size_batched, max_len


    def _create_dataset(self, inputs, targets, batch_size:int = 64, repetitions:int = 100):

        '''creates cached and prefetched datasets from 'inputs' and 'targets' lists'''

        # Create dataset from inputs and targets
        dataset = tf.data.Dataset.from_tensor_slices((
            tf.keras.preprocessing.sequence.pad_sequences(inputs), 
            tf.keras.preprocessing.sequence.pad_sequences(targets)))
        # cache the dataset to memory to get a speedup while reading from it.
        dataset = dataset.cache()
        # create batched dataset and shuffle it
        buffer_size = len(dataset)
        dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True
                    ).repeat(repetitions).padded_batch(batch_size, drop_remainder=True)
        # This allows later elements to be prepared while the current is being processed.
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return buffer_size, dataset


    ############################################################################
    #####################       SAVE AND LOAD           ########################
    ############################################################################

    def save(self, path:str, verbose:bool=False):

        '''saves the dataloader's attributes to a pickle file'''

        if path.endswith('/'):
            filename = path+f'dataloader_{self.comedy_name}_{self.tokenization}.pkl'
        else:
            filename = path+f'/dataloader_{self.comedy_name}_{self.tokenization}.pkl'

        temp = self.datasets.copy()
        self.datasets = None
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        if verbose: print(">> Dataloader saved to:", filename)
        self.datasets = temp

    def load(self, path:str, verbose:bool=False):

        '''initializes the dataloader's attributes from existing pickle file'''

        if path.endswith('/'):
            filename = path+f'dataloader_{self.comedy_name}_{self.tokenization}.pkl'
        else:
            filename = path+f'/dataloader_{self.comedy_name}_{self.tokenization}.pkl'

        with open(filename, 'rb') as f:
            temp = pickle.load(f)
            for attr, value in temp.__dict__.items():
                self.__dict__[attr] = value

        if verbose: print(">> Dataloader loaded from:", filename)
        self._init_datasets()

    ############################################################################
    ##########################          UTILS         ##########################
    ############################################################################

    def print_vocab_info(self):

        '''prints the vocabulary information in user-friendly format'''

        print("\n>> VOCABULARY:")
        for info in self.vocab_info:
            print(" - {}: {}".format(info, self.vocab_info[info]))

    def print_comedy_samples(self, n:int=1):

        '''prints a decoded example of input-target couple for generator's training'''

        # Print samples of the generated Comedy dataset
        for (batch, (inputs, targets)) in enumerate(self.datasets['comedy'].take(n)):
            print("\n{} [ Dataset Sample: {} ] {}\n".format("="*13, batch+1, "="*13))
            print("-- input:\n\n{}\n-- target:\n\n{}".format(
                clear_text(ints_to_text(inputs[0], self.idx2str)),
                clear_text(ints_to_text(targets[0], self.idx2str))
            ))
        print("{}".format("="*45))

    def get_comedy_start(self):
        '''returns the list of the first three verses of the divine comedy'''
        return self.files_dict[self._get_tokenized_filename(self.comedy_name)][:3]

    ############################################################################
    #######################     MANAGE FILENAMES      ##########################
    ############################################################################


    def _get_tokenized_filename(self, filename:str):

        '''returns tokenized version of filename'''

        filename = f"tokenized_{filename}_{self.tokenization}"
        if not ".txt" in filename:
            filename = filename+".txt"
        return filename

    def _get_original_filename(self, filename:str):

        '''returns original version of tokenized_filename'''

        filename = filename.replace(f"_{self.tokenization}", "")
        if not ".txt" in filename:
            filename = filename+".txt"
        if "tokenized" in filename:
            filename.replace("tokenized_", "")
        return filename

    def _get_clean_filename(self, filename:str):

        '''returns clean version of filename'''

        if "tokenized" in filename:
            filename = self._get_original_filename(filename)
        return filename.replace(".txt", "")