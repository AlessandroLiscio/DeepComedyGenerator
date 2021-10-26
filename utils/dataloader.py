from utils.preprocessing import *
import pickle

class DataLoader():

    def __init__(self,
                from_pickle:str = None,
                sep:str = '|',
                in_path:str = 'data/tokenized/',
                comedy_name:str = 'comedy',
                tokenization:str=None,
                epochs_production:int = 0,
                epochs_comedy:int = 0,
                verbose:str = False):

        if from_pickle:
            self.load(from_pickle, verbose)
        else:
            self.comedy_name = comedy_name
            self.sep = sep
            self.tokenization = tokenization
            self.epochs_production = epochs_production
            self.epochs_comedy = epochs_comedy

            self._init_train_order(verbose)
            self._read_files(in_path, verbose)
            self._init_vocab_and_mappings(verbose)
            self._init_datasets(verbose)

    def _get_tokenized_filename(self, original_filename:str):

        '''returns tokenized version of filename'''

        if not self.tokenization:
            return f"tokenized_{original_filename}.txt"
        else:
            return f"tokenized_{original_filename}_{self.tokenization}.txt"

    def _get_original_filename(self, tokenized_filename:str):

        '''returns original version of tokenized_filename'''

        if self.tokenization:
            tokenized_filename = tokenized_filename.replace(f"_{self.tokenization}", "")
        return tokenized_filename.replace("tokenized_", "")

    def _get_clean_filename(self, filename:str):

        '''returns clean version of filename'''

        if "tokenized" in filename:
            filename = self._get_original_filename(filename)
        return filename.replace(".txt", "")

    def _init_train_order(self, verbose:bool=False):

        '''initializes the generator training order, based on the
        training epochs for the production and comdedy datasets'''

        train_order = []
        if self.epochs_production > 0:
            for filename in ['convivio','vita', 'detto','fiore']:
                train_order.append(self._get_tokenized_filename(filename))
        if self.epochs_comedy > 0:
            train_order.append(self._get_tokenized_filename(self.comedy_name))
        self.train_order = train_order

        if verbose:
            print("\n>> TRAINING ORDER:")
            for i, filename in enumerate(self.train_order):
                print(" {}- {}".format(i+1, self._get_clean_filename(filename)))

    def _read_files(self, directory:str='data/', verbose:bool=False):

        '''reads all .txt files in 'directory' and initializes the files
        dictionary, with files names as keys and texts as values'''

        files_list = []
        files_names = []

        if verbose:
            print("\n>> LOADING FILES:")

        for i, filename in enumerate(self.train_order):

            # Read file
            path_to_file = os.path.join(directory, filename)
            text = open(path_to_file, 'r').read().lower()

            # generate list from text by splitting lines
            text_list = text.splitlines()
            files_list.append(text_list)
            files_names.append(filename)

            if verbose:
                print(" {}- {}".format(i+1, path_to_file))

        self.files_dict = {files_names[i]:files_list[i] for i in range(len(files_names))}


    def _init_vocab_and_mappings(self, verbose:bool=False):

        '''initialize vocabularies and mappings'''

        #############################
        # final vocabulary order:   #
        # - special tokens          #
        # - punctuation             #
        # - non-ending syllables    #
        # - ending syllables        #
        #############################

        syllables = sorted(
            set().union(*[verses_to_syllables_set(self.files_dict[file_name], self.sep) for file_name in self.train_order]),
            key=len)

        # initialize groups
        special_tokens = []
        punctuation = []
        non_ending_sylls = []
        ending_sylls = []

        for token in syllables:
            # NON-SYLLABLES
            if '<' in token:
                special_tokens.append(token)
            elif len(token) == 1 and not token.isalpha():
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
        self.vocab = []
        for group in [special_tokens, punctuation, non_ending_sylls, ending_sylls]:
            self.vocab.extend(group)

        # insert the empty string at poistion 0
        if '' in self.vocab:
            self.vocab.remove('')
        self.vocab.insert(0, '')

        # store vocabulary information
        self.vocab_info = {
            'size' : len(self.vocab),
            'syllables start' : len(special_tokens) + len(punctuation) + 1,
            'special tokens' : len(special_tokens),
            'punctuation' : len(punctuation),
            'non-ending syllables' : len(non_ending_sylls),
            'ending syllables' : len(ending_sylls)
        }

        # Creating a mapping from unique characters to indices
        self.str2idx = {u:i for i, u in enumerate(self.vocab)}
        self.idx2str = np.array(self.vocab)

        if verbose:
            print("\n>> VOCABULARY:")
            for info in self.vocab_info:
                print("> {}: {}".format(info, self.vocab_info[info]))

    def _init_datasets(self, verbose:bool=False):
        
        # TODO: remove original_length_"dataset"
        # TODO: remove tercet_max_len

        # Initialize datasets dictionary
        self.datasets = {'production': None,
                        'comedy': None}
        self.original_length_production = 0
        self.original_length_comedy = 0

        ## Production dataset
        if self.epochs_production > 0:
            self.datasets['production'] = []
            for file_name in self.train_order:
                if not file_name == self._get_tokenized_filename(self.comedy_name):
                    self.datasets['production'].append(self.files_dict[file_name])

            # Split input target for Dante's Production dataset
            if verbose: print("\n>> Generating Dante's Production Dataset")
            self.datasets['production'], self.original_length_production = split_input_target_production(
                self.datasets['production'], 
                self.str2idx, 
                sep = self.sep,
                inp_len = 3, tar_len = 3,
                repetitions = self.epochs_production)
            if verbose: print("> Real size production: ", self.original_length_production)

        ## Comedy dataset
        if self.epochs_comedy > 0:
            if verbose: print("\n>> Generating Divine Comedy Dataset")
            self.datasets['comedy'], self.tercet_max_len, self.original_length_comedy = split_input_target_comedy(
                self.files_dict[self._get_tokenized_filename(self.comedy_name)],
                self.str2idx,
                sep = self.sep,
                inp_len = 3, tar_len = 4,
                repetitions = self.epochs_comedy)
            if verbose: print("> Real size comedy: ", self.original_length_comedy)

    def print_comedy_samples(self, n:int=1):

        '''prints an example of input-target couple for generator's training'''

        # Print samples of the generated Comedy dataset
        for (batch, (inputs, targets)) in enumerate(self.datasets['comedy'].take(n)):
            print("\n{} [ Dataset Sample: {} ] {}\n".format("="*13, batch+1, "="*13))
            print("-- input:\n\n{}\n-- target:\n\n{}".format(
                clear_text(ints_to_text(inputs[0], self.idx2str)),
                clear_text(ints_to_text(targets[0], self.idx2str))
            ))
        print("{}".format("="*45))

    def load(self, path:str, verbose:bool=False):

        if path.endswith('/'):
            filename = path+f'dataloader_{self.tokenization}.pkl'
        else:
            filename = path+f'/dataloader_{self.tokenization}.pkl'

        with open(filename, 'rb') as f:
            temp = pickle.load(f)
            for attr, value in temp.__dict__.items():
                self.__dict__[attr] = value
        if verbose: print(">> Dataloader loaded from:", path)
        self._init_datasets(verbose)

    def save(self, path:str, verbose:bool=False):

        if path.endswith('/'):
            filename = path+f'dataloader_{self.tokenization}.pkl'
        else:
            filename = path+f'/dataloader_{self.tokenization}.pkl'

        temp = self.datasets.copy()
        self.datasets = None
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        if verbose: print(">> Dataloader saved to:", path)
        self.datasets = temp

