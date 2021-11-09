# Generator imports
import time
import tensorflow as tf
import numpy as np
from src.transformer import Transformer, create_masks
from src.tokensprocessing import *
import sys

# Results imports
import os
import json
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

_train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
]


class Generator():

    def __init__(self, dataloader,
                 encoders: int = 5, decoders: int = 5, heads: int = 4,
                 d_model: int = 256, dff: int = 512, dropout: float = 0.2,
                 epochs_production: int = 0, epochs_comedy: int = 0,
                 stop = ['</v>'],
                 verbose: bool = True):

        # initialize epochs
        self.epochs = {'production': 0, 'comedy': 0}

        # model instantiation
        self.dataloader = dataloader
        self.model = Transformer(num_layers_encoder=encoders,
                                 num_layers_decoder=decoders,
                                 d_model=d_model,
                                 num_heads=heads,
                                 dff=dff,
                                 input_vocab_size=self.dataloader.vocab_info['size'],
                                 target_vocab_size=self.dataloader.vocab_info['size'],
                                 pe_input=self.dataloader.vocab_info['size'],
                                 pe_target=self.dataloader.vocab_info['size'],
                                 dropout=dropout)

        # optimizer
        self.lr = CustomSchedule(self.model.d_model)
        self.optimizer = tf.keras.optimizers.Adam(self.lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        # training metrics definition
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        # generation_step stop tokens
        self.stop = [self.dataloader.str2idx[stopper] for stopper in stop]

        # generator info
        self._init_log()
        if verbose: print(self)

    def __str__(self):
        return "\n".join((
            "",
            ">> GENERATOR:",
            "> MODEL",
            f" - encoders: {self.model.num_layers_encoder}",
            f" - decoders: {self.model.num_layers_decoder}",
            f" - heads: {self.model.num_heads}",
            f" - d_model: {self.model.d_model}",
            f" - dff: {self.model.dff}",
            f" - dropout: {self.model.dropout}",
            f"> TRAINING",
            f" - optimizer: {str(type(self.optimizer))[:-2].split('.')[-1]}",
            f" - loss: {str(type(self.loss_object))[:-2].split('.')[-1]}",
            f" - metric: {str(type(self.train_accuracy))[:-2].split('.')[-1]}",
            f" - epochs_production: {self.epochs['production']}",
            f" - epochs_comedy: {self.epochs['comedy']}",
            ""
        ))

    ############################################################################
    ######################            TRAINING            ######################
    ############################################################################

    def train_model(self, checkpoint: int = None, out_path: str = None):
        '''train model on all the datasets'''
        for key, dataset in self.dataloader.datasets.items():
            if self.dataloader.repetitions[key] > 0:
                self.train_on_dataset(key, dataset, out_path, checkpoint)

    def train_on_dataset(self,
                         dataset_name,
                         dataset,
                         out_path: str = None,
                         checkpoint: int = None):

        '''train model on target dataset. As the dataset has been
        repeated several times (each time singularly shuffled)
        instead of using the concept of "epochs" on the same dataset,
        the original dataset length is required as input, in order to
        update training metrics and histories at each dataset
        repetition (which basically counts as an "epoch").
        Based on 'dataset_name', at each epoch, the generator parameters
        'epochs_production' or 'epochs_comedy' are updated.'''

        assert (dataset_name == "comedy" or dataset_name == "production")

        # initialize training variables
        start = time.time()
        epoch = 1
        loss_history = []
        accuracy_history = []
        epochs = self.dataloader.repetitions[dataset_name]
        original_length = self.dataloader.original_lengths[dataset_name]

        for (batch, (inp, tar)) in enumerate(dataset):

            # update gradients
            self._train_step(inp, tar)

            # show/update output progress bar
            print_progress(batch, len(dataset),
                           "  ".join((
                               f"epoch {epoch}/{epochs}",
                               "loss: {:.4f}".format(self.train_loss.result()),
                               "accuracy: {:.4f}".format(self.train_accuracy.result()))))

            # At the end of each epoch:
            if batch != 0 and batch % original_length == 0:

                # Increment epoch and restart timer
                self.epochs[dataset_name] += 1
                epoch += 1

                # Update histories
                t = round(time.time() - start)
                self._update_history(dataset_name, t)

                # Reset loss and accuracy states
                self.train_loss.reset_states()
                self.train_accuracy.reset_states()

                # Save model weights, log and history plots every "checkpoint" epochs
                if checkpoint and out_path:
                    if batch != 0 and batch % (original_length * checkpoint) == 0:
                        self.save(out_path, verbose=False)

                start = time.time()

        # Stop timer
        t = self.log["trainings"][dataset_name]["time"]
        print(f'\n\tTraining completed in {int(t / 3600)}h {int(t / 60 % 60)}m {int(t % 60)}s.\n')

        # Update histories and save model weights, log and history plots
        if out_path:
            self.save(out_path, verbose=False)

    @tf.function(input_signature=_train_step_signature)
    def _train_step(self, inp, tar):

        '''single training step: given an input list of verses,
        the model tries to predict the next one. Then loss
        and accuracies are computed and gradients are applied'''

        pred_size = self.dataloader.pred_size

        # split input and target
        tar_inp = tar[:, :-pred_size]
        tar_real = tar[:, pred_size:]

        # create masks
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

        # compute predictions
        with tf.GradientTape() as tape:
            predictions, _ = self.model(inp,
                                        tar_inp,
                                        True,
                                        enc_padding_mask,
                                        combined_mask,
                                        dec_padding_mask
                                        )

            # compute loss function
            loss = self._loss_function(tar_real, predictions)

        # compute gradients and apply gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # update training metrics
        self.train_loss(loss)
        self.train_accuracy(tar_real, predictions)

    def _loss_function(self, real, pred):

        '''model's loss function: given as input the real target
        and the prediction, it computes the loss value ignoring
        the masked tokens.'''

        ######### DEFAULT ########

        # "mask" is a boolean tensor with False values on padding values (0 values) 
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        # "loss_" is a tensor of float values
        loss_ = self.loss_object(real, pred)
        # convert mask boolean values to float (False=0. and True=1.)
        mask = tf.cast(mask, dtype=loss_.dtype)
        # apply mask to loss tensor
        loss_ *= mask

        ######### EOV MASK #########

        # eov mask
        eov_mask = tf.math.equal(real, self.dataloader.eov)
        eov_mask = tf.where(eov_mask, 5.0, 1.0)
        eov_mask = tf.cast(eov_mask, dtype=loss_.dtype)

        # apply mask to loss tensor
        loss_ *= eov_mask

        # returns a single float value representing the loss value
        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

        ########## SYLLS SCORE ##########

        # # real target syllables
        # real_sylls = tf.math.greater_equal(real, self.alphas_start)
        # real_sylls = tf.cast(real_sylls, dtype=loss_.dtype)
        # real_sylls = tf.reduce_sum(real_sylls)

        # # predicted target syllables
        # pred_sylls = tf.math.greater_equal(pred, self.alphas_start)
        # pred_sylls = tf.cast(pred_sylls, dtype=loss_.dtype)
        # pred_sylls = tf.reduce_sum(pred_sylls)

        # # compute syllables score
        # sylls_score = abs(real_sylls - pred_sylls) / real_sylls

        # sylls_score = real_sylls / (real_sylls - abs(real_sylls - pred_sylls) )
        # return tf.reduce_sum(loss_) / tf.reduce_sum(mask) * sylls_score        

    ############################################################################
    ##################            SAVE AND LOAD           ######################
    ############################################################################   

    def save(self, path: str, verbose: bool = True):
        self.save_model_weights(path, verbose)
        self.save_log(path, verbose)
        self.plot_history(path, verbose)

    def load(self, path: str, verbose: bool = True):
        self.load_model_weights(path, verbose)
        self.load_log(path, verbose)

    def save_model_weights(self, path: str, verbose: bool = True):

        '''saves the weights of the model to target path. The name
        of the file is based on the instantiated model's parameters'''

        # create output folders
        create_folder(path)
        path += self.get_dataloader_name() + "/"
        create_folder(path)
        path += self.get_model_name() + "/"
        create_folder(path)
        path += self.get_checkpoint_name() + "/"
        create_folder(path)
        path += "weights/"
        create_folder(path)

        # save weights
        self.model.save_weights(path)
        path = path.replace("weights/", "")
        if verbose: print("\n> Saved weights to checkpoint", path)

    def load_model_weights(self, path: str, verbose: bool = True):

        '''loads the weights of the model from input path, based on the
        instantiated model's parameters'''

        path = self.get_model_folder(path) + "weights/"
        self.model.load_weights(path)
        path = path.replace("weights/", "")
        if verbose: print("\n> Loaded weights from checkpoint", path)

    def load_log(self, path: str, verbose: bool = True):

        '''loads the generator's log from input path, based on the
        instantiated model's parameters'''

        log_path = self.get_model_folder(path) + "log.json"
        with open(log_path, "r") as f:
            self.log = json.load(f)

    ############################################################################
    ######################          GENERATION              ####################
    ############################################################################

    def generate_from_tercet(self, tercet, temperatures, n_verses: int = 100, generation_type:str='sampling'):

        '''generates 'n_verses' for each temperature, starting from
        input tercet, where every verse has at most 'tercet_max_len' tokens'''

        # prepare input tercet in order to feed it to the model
        start = list(tf.keras.preprocessing.sequence.pad_sequences(
            [flatten(
                encode_tokens(
                    split_tokens(tercet, self.dataloader.separator),
                    self.dataloader.str2idx))],
            maxlen=self.dataloader.tercet_max_len,
            padding='post')[0])

        # print("\nStart:\n", np.array(tercet))        
        print("\nGenerating new cantica: ")

        # generate a cantica for each temperature
        generations = []

        for temp in temperatures:
            print(f"- temperature {temp}... ", end="")

            # start timer
            t_start = time.time()

            # generate cantica
            generated = self._generate(start = start,
                                        max_len = self.dataloader.tercet_max_len,
                                        n_verses = n_verses,
                                        temperature = temp,
                                        generation_type = generation_type)

            # decode the generated cantica and remove special tokens
            generated = clear_text(ints_to_text(generated, self.dataloader.idx2str))

            # append generated cantica to results
            generations.append(generated)

            # stop timer
            t_gen = round(time.time() - t_start)
            print(f"completed ({int(t_gen / 3600)}h {int(t_gen / 60 % 60)}m {int(t_gen % 60)}s)")

        self.log["generations"] = {}
        for i, temp in enumerate(temperatures):
            self.log["generations"]["temp_" + str(temp)] = generations[i]

        return self.log["generations"]

    def _generate(self, start, max_len: int = 100, n_verses: int = 100, temperature: int = 1.0, generation_type:str='sampling'):

        '''generates 'n_verses' verses, starting from input 'start', where every 
        verse has at most 'max_len' tokens. The generation probability is 
        influenced by the temperature: the higher the temperature, the more 
        original (or crazy) is the text.'''

        # updates_input_sequence based on "stop" tokens
        def update_input_sequence(sequence):
            for i, token in enumerate(sequence):
                if token in self.stop:
                    return sequence[i+1:]

        # variables initialization
        input_sequence = start.copy()
        output = []

        # if generation_type == 'beam_search':
        #     n_verses = int(n_verses / 3) + 1

        try:

            for _ in range(n_verses):

                # pad the input list to reach the max_len
                input_sequence = list(
                    tf.keras.preprocessing.sequence.pad_sequences(
                        [input_sequence],
                        maxlen=max_len)[0])

                # print(clear_text(ints_to_text(input_sequence, self.dataloader.idx2str)))

                # generate one verse
                generated, _ = self._generation_step(input_sequence,
                                                    max_len = max_len,
                                                    temperature = temperature,
                                                    generation_type = generation_type)

                # print('\n', len(generated))
                # print(generated)
                # print(clear_text(ints_to_text(generated, self.dataloader.idx2str)))

                # update the input sequence
                if self.dataloader.pred_size == 1:
                    input_sequence += generated
                    input_sequence = update_input_sequence(input_sequence)
                else:
                    input_sequence = generated

                if input_sequence == None:
                    return output

                # append the generated verse to the output
                output += generated

        except e:
            print("ERROR: ", e)
            pass

        return output

    def _generation_step(self, input_sequence, max_len:int=100, temperature:int=1.0, generation_type:str='sampling'):

        '''generate tokens, starting from 'input_sequence'.'''

         # add the batch dimension for compatibility
        encoder_input = tf.expand_dims(input_sequence, 0)
        decoder_input = tf.expand_dims(input_sequence, 0)

        if generation_type == 'sampling':
            
            # the final output of the evaluation (initially, this is an empty list)
            output = []

            # we repeat the process to get the entire verse (end-of-verse token is predicted)
            # for i in range(int(max_len/3)):
            for i in range(max_len):

                enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, decoder_input)
                logits, attention_weights = self.model(
                    encoder_input, decoder_input, False,
                    enc_padding_mask, combined_mask, dec_padding_mask
                )
            
                # the higher the temperature, the more original (or crazy) is the text
                predictions = logits[: ,:, :]
                predictions /= temperature
                predicted_id = tf.cast(tf.random.categorical(tf.squeeze(predictions, 0), num_samples=1)[-1,0].numpy() , tf.int32)
            
                # append the predicted token to the output
                output.append(predicted_id.numpy())
            
                # stop generation if the token coincides one of the "stop" tokens
                if predicted_id in self.stop: break
            
                # otherwise the token is appended both to the new decoder input
                decoder_input = tf.concat([decoder_input, [[predicted_id]]], axis=-1)
            
        elif generation_type == 'beam_search':
            
            beams, attention_weights = self.beam_search_decoder(encoder_input,
                                                                decoder_input,
                                                                max_len,
                                                                beam_width=5,
                                                                verbose=False)

            output = beams[0][0]

        else:
            raise ValueError(f"Incorrect 'generation_type' parameter found.")
            return None, None

        return output, attention_weights

    ############################################################################
    ######################          BEAM SEARCH          #######################
    ############################################################################

    def beam_search_decoder(self, encoder_input, decoder_input, max_len:int, beam_width:int=5, verbose:bool=False):

        tokens, probabilities, attention_weights = self._beam_search_decoding_step(encoder_input, decoder_input, beam_width)
        beams = [[[token], prob] for (token, prob) in zip(tokens, probabilities)]
        ended = []
        if verbose:
            print(f'\nbeams: {beams}')

        for i in range(max_len):
            candidates = []

            for j, [beam, prob] in enumerate(beams):
                if not beam[-1] in self.stop:
                    tokens_temp, probabilities_temp, attention_weights = self._beam_search_decoding_step(
                        encoder_input,
                        tf.concat([decoder_input, [tf.cast(beam, tf.int32)]], axis=-1),
                        beam_width)
                    for token, prob_temp in zip(tokens_temp, probabilities_temp):
                        candidates.append([beam + [token], prob + prob_temp])
                else:
                    ended.append([beam, prob])
                    if len(ended) == beam_width:
                        return sorted(ended, reverse=True, key=lambda x: x[1]), attention_weights
            beams = sorted(candidates, key=lambda x: x[1])[-(beam_width - len(ended)):]
            if verbose:
                print(f'i: {i}\n\tbeams: {beams}\n\tended: {ended}\n')
        return sorted(ended, reverse=True, key=lambda x: x[1]), attention_weights

    def _beam_search_decoding_step(self, encoder_input, decoder_input, beam_width):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, decoder_input)
        logits, attention_weights = self.model(
            encoder_input, decoder_input, False,
            enc_padding_mask, combined_mask, dec_padding_mask
        )
        predictions = logits[:, :, :]
        predictions = tf.nn.softmax(predictions, axis=-1)  # softmax
        np.set_printoptions(precision=3, threshold=sys.maxsize)
        # select last token's logits
        predictions = tf.squeeze(predictions, 0)[-1, :].numpy()
        predictions = np.log(predictions)
        tokens = np.argsort(predictions)[-beam_width:]
        probabilities = predictions[tokens]
        return tokens, probabilities, attention_weights

    ############################################################################
    ######################          GENERATOR INFO            ##################
    ############################################################################     

    def _init_log(self):

        '''initializes the model's log dictionary'''

        self.log = {
            "model": {
                "encoders": self.model.num_layers_encoder,
                "decoders": self.model.num_layers_decoder,
                "heads": self.model.num_heads,
                "d_model": self.model.d_model,
                "dff": self.model.dff,
                "droupout": self.model.dropout
            },
            "dataloader": {
                "comedy_name": self.dataloader.comedy_name,
                "tokenization": self.dataloader.tokenization,
                "separator": self.dataloader.separator,
                "original_length_production": self.dataloader.original_lengths['production'],
                "original_length_comedy": self.dataloader.original_lengths['comedy'],
                "tercet_max_len": self.dataloader.tercet_max_len,
                "train_order": self.dataloader.train_order,
                "vocab_info": self.dataloader.vocab_info,
            },
            "trainings": {
                "info": {
                    "optimizer": str(type(self.optimizer))[:-2].split('.')[-1],
                    "loss": str(type(self.loss_object))[:-2].split('.')[-1],
                    "metric": str(type(self.train_accuracy))[:-2].split('.')[-1]
                },
                "production": {
                    "epochs": self.epochs['production'],
                    "time": 0,
                    "loss_history": [],
                    "acc_history": []
                },
                "comedy": {
                    "epochs": self.epochs['comedy'],
                    "time": 0,
                    "loss_history": [],
                    "acc_history": []
                }
            }
        }

    def _update_history(self, dataset_name, t):

        '''update log training histories'''

        self.log["trainings"][dataset_name]["epochs"] = self.epochs[dataset_name]
        self.log["trainings"][dataset_name]["time"] += t

        self.log["trainings"][dataset_name]["loss_history"].append("{:.4f}".format(self.train_loss.result()))
        self.log["trainings"][dataset_name]["acc_history"].append("{:.4f}".format(self.train_accuracy.result()))

    def get_dataloader_name(self):
        return self.dataloader.get_name()

    def get_model_name(self):
        return "_".join((
            f"{self.model.num_layers_encoder}",
            f"{self.model.num_layers_decoder}",
            f"{self.model.num_heads}",
            f"{self.model.d_model}",
            f"{self.model.dff}"
        ))

    def get_checkpoint_name(self):
        return f"{self.epochs['production']}_{self.epochs['comedy']}"

    def get_model_folder(self, out_path: str):
        return f"{out_path}{self.get_dataloader_name()}/{self.get_model_name()}/{self.get_checkpoint_name()}/"

    ############################################################################
    #################               RESULTS                #####################
    ############################################################################

    def print_training_info(self):

        '''print training information'''

        print('\n>> TRAINING INFO:')
        for training in self.log['trainings']:
            print(f" -- {training}")
            for info in self.log['trainings'][training]:
                if 'history' in info:
                    hist = self.log['trainings'][training][info]
                    print(f"   -- {info}: {hist[:3]} ... {hist[-3:]}")
                elif info == 'time':
                    t = self.log['trainings'][training][info]
                    print(f"   -- {info}: {int(t / 3600)}h {int(t / 60 % 60)}m {int(t % 60)}s")
                else:
                    print(f"   -- {info}: {self.log['trainings'][training][info]}")

    def plot_history(self, out_path: str = None, verbose: str = True):

        '''plot loss and accuracy histories. If 'out_path' is not None
        the figure is saved in 'out_path'. If 'verbose' is True the
        plot is shown.'''

        for dataset_name in self.log['trainings']:

            if dataset_name != 'info':

                fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(30, 10))

                # loss history
                loss_history = self.log['trainings'][dataset_name]['loss_history']
                for i, loss in enumerate(loss_history):
                    loss_history[i] = float(loss_history[i])

                # accuracy history
                acc_history = self.log['trainings'][dataset_name]['acc_history']
                for i, loss in enumerate(acc_history):
                    acc_history[i] = float(acc_history[i])

                # plot loss history
                ax0.set_title('Loss History', color='lightblue', fontsize=15, fontweight='bold')
                ax0.set_xticks(range(0, len(loss_history), 5))
                ax0.grid()
                ax0.plot(loss_history, color='blue')

                # plot accuracy history
                ax1.set_title('Accuracy History', color='orange', fontsize=15, fontweight='bold')
                ax1.set_xticks(range(0, len(acc_history), 5))
                ax1.set_ylim(top=1)
                ax1.grid()
                ax1.plot(acc_history, color='red')

                # save plot as .png and show it
                if out_path:
                    plt.savefig(self.get_model_folder(out_path) + f"history_{dataset_name}.png")

                # if verbose, display plot
                if verbose:
                    plt.show()

                # close figure
                plt.close()

    def generations_table(self, out_path: str = None, verbose: bool = True):

        '''print generations in tabular form'''

        # Generations
        generations = []
        max_len = 0
        for temp in self.log['generations']:
            canto = self.log['generations'][temp]
            canto = canto.replace(' ,', ',')
            canto = canto.replace(' .', '.')
            canto = canto.replace(' !', '!')
            canto = canto.replace(' ?', '?')
            canto = canto.replace(' :', ':')
            canto = canto.replace(' ;', ';')
            canto = canto.split('\n')
            generations.append(canto)
            if len(self.log['generations'][temp]) > max_len:
                max_len = len(self.log['generations'][temp])

        # header of the table
        head_line = "\n\t    "
        temperatures = [temp for temp in self.log['generations']]
        for temp in temperatures:
            head_line += "{:<45}".format(temp)
        head_line += "\n\n"

        # organize by columns
        rows = []
        for row_idx in range(0, max_len):
            row = ""
            for temp in range(0, len(temperatures) - 1):
                if row_idx >= len(generations[temp]):
                    row += " " * 45
                else:
                    row += "{:<45}".format(generations[temp][row_idx])
            rows.append(row)

        # print out
        if verbose:
            print(head_line)
            for row in rows:
                print(row)

        # save table to file
        if out_path:
            with open(self.get_model_folder(out_path) + "generations_table.txt", "w+") as f:
                f.write(head_line)
                for row in rows:
                    f.write(row + '\n')

    def save_log(self, out_path: str, verbose: bool = True):

        '''save log dictionary as .json'''

        # Save the log file
        out_path = self.get_model_folder(out_path) + "log.json"
        with open(out_path, 'w+') as f:
            json.dump(self.log, f, indent=4)
            if verbose: print(f"\n> Log saved in {out_path}")

    def save_generations(self, out_path:str, generation_type:str, verbose:bool=True):

        '''save log dictionary as .json file and generations
        texts as .txt files in 'out_path' folder'''

        # create generations folder if it doesn't exist
        out_path = self.get_model_folder(out_path) + "generations/"
        create_folder(out_path)
        out_path += f"{generation_type}/"
        create_folder(out_path)

        # extract the texts from the log
        generations = []
        for temp in self.log['generations']:
            canto = self.log['generations'][temp]
            canto = canto.replace(' ,', ',')
            canto = canto.replace(' .', '.')
            canto = canto.replace(' !', '!')
            canto = canto.replace(' ?', '?')
            canto = canto.replace(' :', ':')
            canto = canto.replace(' ;', ';')
            canto = canto.split('\n')
            generations.append(canto)

        # Save the generations as text files
        generations_files = []
        for temperature, generated_text in zip(self.log["generations"], generations):
            file_name = f"GEN-{temperature}.txt"
            file_path = out_path + file_name
            with open(file_path, "w+") as out_file:
                out_file.write("\n".join(generated_text[1:]))
                generations_files.append(file_path)

        if verbose: print(f"> Generations saved in folder {out_path}")


############################################################################
#################                UTILS                 #####################
############################################################################

def print_progress(done: int, total: int, *args):
    '''prints model training progress'''

    maxlen = 25
    bars = round(done * maxlen / total)
    print("\r[{}{}] {:3}%".format("=" * int(bars),
                                  " " * int((maxlen - bars)),
                                  round(done * 100 / total)),
          end="\t {:>5}/{:<5}\t{}\t".format(done,
                                            total,
                                            [str(a) for a in args]))


def create_folder(path: str):
    '''create folder if it doesn't exist'''

    if not os.path.exists(path):
        os.mkdir(path)


############################################################################
##############          CUSTOM OPTIMIZER'S SCHEDULE          ###############
############################################################################

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    '''custom schedule class for computing model learning rate'''

    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
