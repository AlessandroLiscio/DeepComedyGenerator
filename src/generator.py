# Generator imports
import time
import tensorflow as tf
from src.transformer import Transformer, create_masks
from src.dataprocessing import *

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
                    encoders:int = 5, decoders:int = 5, heads:int = 4,
                    d_model:int = 256, dff:int = 512, dropout:float = 0.2,
                    epochs_production:int = 0, epochs_comedy:int = 0,
                    verbose:bool = True):  
        
        # initialize trained epochs
        self.trained_epochs_production = 0
        self.trained_epochs_comedy = 0

        # transformer model instantiation
        self.dataloader = dataloader
        self.model = Transformer(num_layers_encoder = encoders,
                                num_layers_decoder = decoders,
                                d_model = d_model,
                                num_heads = heads,
                                dff = dff,
                                input_vocab_size =self.dataloader.vocab_info['size'],
                                target_vocab_size = self.dataloader.vocab_info['size'],
                                pe_input = self.dataloader.vocab_info['size'], 
                                pe_target = self.dataloader.vocab_info['size'],
                                dropout = dropout)

        # optimizer
        self.lr = CustomSchedule(self.model.d_model)
        self.optimizer = tf.keras.optimizers.Adam(self.lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        # training metrics definition
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

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
            f" - epochs_production: {self.trained_epochs_production}/{self.dataloader.repetitions_production}",
            f" - epochs_comedy: {self.trained_epochs_comedy}/{self.dataloader.repetitions_comedy}",
            ""
        ))


    ############################################################################
    ######################            TRAINING            ######################
    ############################################################################      
        
    def _loss_function(self, real, pred):

        '''model's loss function: given as input the real target
        and the prediction, it computes the loss value ignoring
        the masked tokens.'''

        # "mask" is a boolean tensor with False values on padding values (0 values) 
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        # "loss_" is a tensor of float values
        loss_ = self.loss_object(real, pred)
        # convert mask boolean values to float (False=0. and True=1.)
        mask = tf.cast(mask, dtype=loss_.dtype)
        # apply mask to loss tensor
        loss_ *= mask

        # # syllables mask
        # sylls_mask = tf.math.greater_equal(pred, self.alphas_start)
        # hendec_score = abs(11.0 - tf.reduce_sum(tf.cast(sylls_mask, tf.float32))/4)
        # # sylls_mask *= tf.where(sylls_mask, 1.2, 1.0)
        # return tf.reduce_sum(loss_)/tf.reduce_sum(mask)*hendec_score
        
        # returns a single float value representing the loss value
        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


    @tf.function(input_signature=_train_step_signature)
    def _train_step(self, inp, tar):
            
        '''single training step: given an input list of verses,
        the model tries to predict the next one. Then loss
        and accuracies are computed and gradients are applied'''

        # split input and target
        pred_size = 1
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
        
        # compute gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)    
        
        # apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # update training metrics
        self.train_loss(loss)
        self.train_accuracy(tar_real, predictions)

    
    def train_on_dataset(self,
                        dataset,
                        dataset_name,
                        weights_path:str = None,
                        checkpoint:int = None):

        '''train model on target dataset. As the dataset has been
        repeated several times (each time singularly shuffled)
        instead of using the concept of "epochs" on the same dataset,
        the original dataset length is required as input, in order to
        update training metrics and histories at each dataset
        repetition (which basically counts as an "epoch").
        Based on 'dataset_name', at each epoch, the generator parameters
        'epochs_production' or 'epochs_comedy' are updated.'''

        assert (dataset_name == "comedy" or dataset_name == "production")

        # start timer
        start = time.time()
        
        # initialize training variables
        epoch = 1
        loss_history = []
        accuracy_history = []
            
        # compute original dataset size
        if dataset_name == 'comedy':
            epochs = self.dataloader.repetitions_comedy
            original_length = self.dataloader.original_length_comedy
        else:
            epochs = self.dataloader.repetitions_production
            original_length = self.dataloader.original_length_production

        for (batch, (inp, tar)) in enumerate(dataset):
                
            # update gradients
            self._train_step(inp, tar)
        
            # show/update output progress bar
            print_progress(
                batch,
                len(dataset),
                "  ".join((
                    f"epoch {epoch}/{epochs}",
                    "loss: {:.4f}".format(self.train_loss.result()),
                    "accuracy: {:.4f}".format(self.train_accuracy.result())
                )))

            # update metrics and training history at each epoch
            if batch != 0 and batch % original_length == 0:

                # Append values to histories
                loss_history.append('{:.4f}'.format(self.train_loss.result()))
                accuracy_history.append('{:.4f}'.format(self.train_accuracy.result()))

                # Reset loss and accuracy states
                self.train_loss.reset_states()
                self.train_accuracy.reset_states()
                epoch +=1

                # Update generator trained epochs
                if dataset_name == "comedy":
                    self.trained_epochs_comedy += 1
                else:
                    self.trained_epochs_production += 1

            # save model weights every 10 epochs
            if checkpoint and weights_path:
                if batch != 0 and batch % (original_length*checkpoint) == 0:
                    self.save_model_weights(weights_path)
        
        # append last values to histories
        loss_history.append('{:.4f}'.format(self.train_loss.result()))
        accuracy_history.append('{:.4f}'.format(self.train_accuracy.result()))

        # stop timer
        t = round(time.time() - start)
        print(f'\n\tTraining completed in {int(t/3600)}h {int(t/60%60)}m {int(t%60)}s.\n')

        # save model weights
        if weights_path:
            self.save_model_weights(weights_path)
        
        return (t, loss_history, accuracy_history)

    def train_model(self, checkpoint:int = None, out_path:str = None):
        
        out_path += "weights/"

        for key, dataset in self.dataloader.datasets.items():

            # train model on single dataset
            if (key == "production" and self.dataloader.repetitions_production > 0) or (
                key == "comedy" and self.dataloader.repetitions_comedy > 0):
                history = self.train_on_dataset(dataset, key, out_path, checkpoint)

                # update generator log
                self.log["trainings"][key]["time"] = history[0]
                self.log["trainings"][key]["loss_history"] = history[1]
                self.log["trainings"][key]["acc_history"] = history[2]
            

    ############################################################################
    ################       SAVE AND LOAD WEIGHTS           #####################
    ############################################################################   

    def save_model_weights(self, path:str):

        '''saves the weights of the model to target path. The name
        of the file is based on the instantiated model's parameters'''

        # get model name for the file name
        model_name = self.get_model_name()

        # create weights folder if it doesn't exist
        create_folder(path)

        # create model's weights folder 
        w_path = path + model_name + "/"
        create_folder(w_path)

        # create model's weight checkpoint folder
        w_path += f"{self.trained_epochs_production}_{self.trained_epochs_comedy}/"
        create_folder(w_path)

        # save weights
        try:
            w_path = w_path+model_name
            self.model.save_weights(w_path)
        except:
            print(f"ERROR: problem saving weights to {w_path}")

    def load_model_weights(self, path:str):

        '''loads the weights of the model from input path, based on the
        instantiated model's parameters'''

        # get model name for the file name
        model_name = self.get_model_name()

        # load the right checkpoint, based on generator 'trained_epochs' parameters.
        # If not found, the weights will be loaded from the checkpoint with the
        # highest number of trained epochs 
        w_path = f"{path}{model_name}/"
        ckpt_path = w_path + f"{self.trained_epochs_production}_{self.trained_epochs_comedy}/"

        try:
            # load model's checkpoint
            if os.path.isdir(ckpt_path):
                self.model.load_weights(ckpt_path+model_name)
                print("Loaded weights from checkpoint ", ckpt_path+model_name)
            else:
                print("".join((
                    "\n>> Weights not found for ",
                    f"trained_epochs_production = {self.trained_epochs_production} and ",
                    f"trained_epochs_comedy = {self.trained_epochs_comedy}.",
                    "\n>> Loading weights from checkpoint with highest number of trained epochs.\n"
                )))

                # find model's checkpoint with the most trained epochs
                max_epochs = 0
                for entry in os.scandir(w_path):
                    if entry.is_dir():
                        ckpt = entry.path.split("/")[-1]
                        epochs_total = 0
                        for epochs in ckpt.split("_"):
                            epochs_total += int(epochs)
                        if epochs_total > max_epochs:
                            max_epochs = epochs_total
                            ckpt_path = entry.path + "/"

                # load weights
                self.model.load_weights(ckpt_path+model_name)
                print(">> Loaded weights from " + ckpt_path)
        except:
            print(f"ERROR: problem loading weights from {w_path}")


    ############################################################################
    ######################          GENERATION              ####################
    ############################################################################

    def _generate_verse(self, input_list, eov, max_len:int=100, temperature:int=1.0):

        '''generate tokens, starting from 'input_list', until 'eov' token
        is generated or 'max_len' tokens limit has been reached. The generation
        probability is influenced by the temperature: the higher the temperature,
        the more original (or crazy) is the text.'''
    
        # add the batch dimension for compatibility
        encoder_input = tf.expand_dims(input_list, 0)
        decoder_input = tf.expand_dims(input_list, 0)
        
        # the final output of the evaluation (initially, this is an empty list)
        output = []
        
        # we repeat the process to get the entire verse (end-of-verse token is predicted)
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
            output.append(predicted_id)
        
            # stop generation if the token coincides with the end-of-verse token
            if predicted_id == eov: break
        
            # otherwise the token is appended both to the new decoder input
            decoder_input = tf.concat([decoder_input, [[predicted_id]]], axis=-1)
        
        return output, attention_weights

    def _generate(self, start, max_len:int=100, n_verses:int=100, temperature:int=1.0):

        '''generates 'n_verses' verses, starting from input 'start', where every 
        verse has at most 'max_len' tokens. The generation probability is 
        influenced by the temperature: the higher the temperature, the more 
        original (or crazy) is the text.'''

        # end-of-verse
        eov = self.dataloader.str2idx['</v>']

        # drop the first verse to keep a window of 3 verses
        def drop_first_verse(sequence):
            for i, token in enumerate(sequence):
                if token == eov:
                    return sequence[i+1:]

        # variables initialization
        input_sequence = start.copy()
        generated = []

        try:
            for _ in range(n_verses):

                # pad the input list to reach the max_len
                input_list = list(
                    tf.keras.preprocessing.sequence.pad_sequences(
                            [input_sequence],
                            maxlen=max_len
                        )[0]
                    )

                # generate one verse
                generated_verse, _ = self._generate_verse(input_list,
                                                        eov = eov,
                                                        max_len = int(max_len/3),
                                                        temperature=temperature)

                # append the generated verse to the input sequence
                input_sequence += generated_verse

                # drop the first verse to keep a window of 3 verses
                input_sequence = drop_first_verse(input_sequence)

                # append the generated verse to the output
                generated += generated_verse
        except:
            return generated
        
        return generated

    def generate_from_tercet(self, tercet, temperatures, n_verses:int=100):

        '''generates 'n_verses' for each temperature, starting from
        input tercet, where every verse has at most 'tercet_max_len' tokens'''

        # prepare input tercet in order to feed it to the model
        start = list(tf.keras.preprocessing.sequence.pad_sequences(
            [flatten(
                encode_tokens(
                    split_tokens(tercet, self.dataloader.separator),
                    self.dataloader.str2idx))],
            maxlen = self.dataloader.tercet_max_len)[0])

        print("\nStart:\n", np.array(tercet))        
        print("\nGenerating new cantica: ")

        # generate a cantica for each temperature
        generations = []
        for temp in temperatures:

            print(f"- temperature {temp}... ", end="")

            # start timer
            t_start = time.time()

            # generate cantica
            generated_string = self._generate(start = start,
                                                max_len = self.dataloader.tercet_max_len,
                                                n_verses = n_verses,
                                                temperature = temp)

            # decode the generated cantica and remove special tokens
            generated_string = clear_text(ints_to_text(generated_string, self.dataloader.idx2str))

            # append generated cantica to results
            generations.append(generated_string)

            # stop timer
            t_gen = round(time.time() - t_start)
            print(f"completed ({int(t_gen/3600)}h {int(t_gen/60%60)}m {int(t_gen%60)}s)")

        self.log["generations"] = {}
        for i, temp in enumerate(temperatures):
            self.log["generations"]["temp_"+str(temp)] = generations[i]
        
        return self.log["generations"]


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
                "droupout" : self.model.dropout
            },
            "dataloader": {
                "comedy_name": self.dataloader.comedy_name,
                "tokenization": self.dataloader.tokenization,
                "separator": self.dataloader.separator,
                "original_length_production": self.dataloader.original_length_production,
                "original_length_comedy": self.dataloader.original_length_comedy,
                "tercet_max_len": self.dataloader.tercet_max_len,
                "train_order": self.dataloader.train_order,
                "vocab_info": self.dataloader.vocab_info,
            },
            "trainings": {
                "info": {
                    "optimizer" : str(type(self.optimizer))[:-2].split('.')[-1],
                    "loss" : str(type(self.loss_object))[:-2].split('.')[-1],
                    "metric" : str(type(self.train_accuracy))[:-2].split('.')[-1]
                },
                "production": {
                    "epochs" : self.trained_epochs_production,
                    "time": 0,
                    "loss_history": ["0"],
                    "acc_history": ["0"]
                },
                "comedy": {
                    "epochs" : self.trained_epochs_comedy,
                    "time": 0,
                    "loss_history": ["0"],
                    "acc_history": ["0"]
                }
            }
        }

    def get_model_name(self):

        '''stringify the model description for the file name'''

        return "_".join((
            f"{self.dataloader.comedy_name}",
            f"{self.dataloader.tokenization}",
            f"{self.model.num_layers_encoder}",
            f"{self.model.num_layers_decoder}",
            f"{self.model.num_heads}",
            f"{self.model.d_model}",
            f"{self.model.dff}",
            f"{self.dataloader.repetitions_production}",
            f"{self.dataloader.repetitions_comedy}"
        ))

    ############################################################################
    #################               RESULTS                #####################
    ############################################################################


    def show_train_info(self):

        '''print training information'''

        print('\nTRAINING INFO:')
        for training in self.log['trainings']:
            print(f" -- {training}")
            for info in self.log['trainings'][training]:
                if 'history' in info:
                    hist = self.log['trainings'][training][info]
                    print(f"   -- {info}: {hist[:3]} ... {hist[-3:]}")
                elif info == 'time':
                    t = self.log['trainings'][training][info]
                    print(f"   -- {info}: {int(t/3600)}h {int(t/60%60)}m {int(t%60)}s")
                else:
                    print(f"   -- {info}: {self.log['trainings'][training][info]}")
                    
    def plot_hist(self, dataset_name:str='comedy', out_path:str=None, verbose:str=True):

        '''plot loss and accuracy histories and save figure
        in 'out_path' folder as 'history.png' file if 'out_path'
        is defined.'''

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
        ax0.set_title('Loss History', color='lightblue', fontsize=15, fontweight= 'bold')
        ax0.set_xticks(range(0,len(loss_history),5))
        ax0.grid()
        ax0.plot(loss_history, color='blue')

        # plot accuracy history
        ax1.set_title('Accuracy History', color='orange', fontsize=15, fontweight= 'bold')
        ax1.set_xticks(range(0,len(acc_history),5))
        ax1.set_ylim(top=1)
        ax1.grid()
        ax1.plot(acc_history, color='red')

        # save plot as .png and show it
        if out_path:
            plt.savefig(out_path+f"{dataset_name}_history.png")
        
        # if verbose, display plot
        if verbose:
            plt.show()

    def tabular_generations(self, out_path:str=None, verbose:bool=True):

        '''print generations in tabular form'''

        # Generations
        generations = []
        temperatures = []
        max_len = 0
        for temp in self.log['generations']:
            canto = self.log['generations'][temp]
            canto = canto.replace(' ,',',')
            canto = canto.replace(' .','.')
            canto = canto.replace(' !','!')
            canto = canto.replace(' ?','?')
            canto = canto.replace(' :',':')
            canto = canto.replace(' ;',';')
            canto = canto.split('\n')
            generations.append(canto)
            temperatures.append(temp)
            if len(self.log['generations'][temp]) > max_len:
                max_len = len(self.log['generations'][temp])

        # header of the table
        head_line = "\n\t    "
        for temp in temperatures:
            head_line += "{:<45}".format(temp)
        head_line += "\n\n"
        
        # organize by columns
        rows = []
        for row_idx in range(0, max_len):
            row = ""
            for temp in range(0, len(temperatures)-1):
                if row_idx >= len(generations[temp]):
                    row += " "*45
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
            with open(out_path+"generations_table.txt", "w+") as f:
                f.write(head_line)
                for row in rows:
                    f.write(row+'\n')

    def save_log(self, out_path:str, verbose:bool=True):

        '''save log dictionary as .json'''

        # stringify the model description for the file name
        model_name = self.get_model_name()

        # Save the log file 
        filename = f"{out_path}LOG_{model_name}.json"
        with open(filename, 'w+') as f:
            json.dump(self.log, f, indent=4)
            if verbose: print(f"log saved as {filename}")

    def save_generations(self, out_path:str, verbose:bool=True):

        '''save log dictionary as .json file and generations
        texts as .txt files in 'out_path' folder'''

        # stringify the model description for the file name
        model_name = self.get_model_name()

        # create generations folder if it doesn't exist
        out_path += "generations/"
        create_folder(out_path)
        out_path += f"{model_name}/"
        create_folder(out_path)

        # extract the texts from the log
        generations = []
        for temp in self.log['generations']:
            canto = self.log['generations'][temp]
            canto = canto.replace(' ,',',')
            canto = canto.replace(' .','.')
            canto = canto.replace(' !','!')
            canto = canto.replace(' ?','?')
            canto = canto.replace(' :',':')
            canto = canto.replace(' ;',';')
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
                if verbose: print(f"generated text at temperature {temperature} saved as {file_name}")
        if verbose: print(f"\tin folder {out_path}")


############################################################################
#################                UTILS                 #####################
############################################################################

def print_progress(done:int, total:int, *args):

  '''prints model training progress'''

  maxlen = 25
  bars = round(done*maxlen/total)
  print("\r[{}{}] {:3}%".format("="*int(bars),
                                " "*int((maxlen - bars)),
                                round(done*100/total)), 
        end="\t {:>5}/{:<5}\t{}\t".format(done,
                                          total,
                                          [str(a) for a in args]))

def create_folder(path:str):

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
