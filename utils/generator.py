import time
import tensorflow as tf
from utils.transformer import Transformer, create_masks
from utils.training import print_progress, CustomSchedule
from utils.results import create_folder
from utils.preprocessing import *

_train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
]

class Generator():

    def __init__(self, dataloader,
                    encoders:int = 5, decoders:int = 5, heads:int = 4,
                    d_model:int = 256, dff:int = 512, dropout:float = 0.2,
                    epochs_production:int = 0, epochs_comedy:int = 0):
    
        # initialize mappings
        self.str2idx = dataloader.str2idx
        self.idx2str = dataloader.idx2str

        # initialize transformer model parameters
        self.tokens_vocab_size = dataloader.vocab_info['size']
        self.encoders = encoders
        self.decoders = decoders
        self.heads = heads
        self.d_model = d_model
        self.dff = dff
        self.dropout = dropout

        # initialize training epochs
        self.epochs_production = dataloader.epochs_production
        self.epochs_comedy = dataloader.epochs_comedy

        # initialize trained epochs
        self.trained_epochs_production = 0
        self.trained_epochs_comedy = 0

        # datasets info
        self.sep = dataloader.sep
        self.tercet_max_len = dataloader.tercet_max_len
        self.original_length_production = dataloader.original_length_production
        self.original_length_comedy = dataloader.original_length_comedy

        # transformer model instantiation
        self.model = Transformer(encoders,
                                decoders,
                                d_model,
                                heads,
                                dff,
                                input_vocab_size=self.tokens_vocab_size,
                                target_vocab_size= self.tokens_vocab_size,
                                pe_input= self.tokens_vocab_size, 
                                pe_target= self.tokens_vocab_size,
                                rate= dropout)

        # optimizer
        self.lr = CustomSchedule(self.d_model)
        self.optimizer = tf.keras.optimizers.Adam(self.lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        # training metrics definition
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    def __str__(self):
        return "\n".join((
            "",
            ">> GENERATOR:",
            f"> encoders: {self.encoders}",
            f"> decoders: {self.decoders}",
            f"> num_heads: {self.heads}",
            f"> d_model: {self.d_model}",
            f"> dff: {self.dff}",
            f"> tokens_vocab_size: {self.tokens_vocab_size}",
            f"> dropout: {self.dropout}",
            f"> optimizer: {str(type(self.optimizer))[:-2].split('.')[-1]}",
            f"> loss: {str(type(self.loss_object))[:-2].split('.')[-1]}",
            f"> metric: {str(type(self.train_accuracy))[:-2].split('.')[-1]}",
            f"> epochs_production: {self.epochs_production}",
            f"> epochs_comedy: {self.epochs_comedy}",
            f"> trained_epochs_production: {self.trained_epochs_production}",
            f"> trained_epochs_comedy: {self.trained_epochs_comedy}",
            ""
        ))

    ##############
    # GENERATION #
    ##############

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
        eov = self.str2idx['</v>']

        # drop the first verse to keep a window of 3 verses
        def drop_first_verse(sequence):
            for i, token in enumerate(sequence):
                if token == eov:
                    return sequence[i+1:]

        # variables initialization
        input_sequence = start.copy()
        generated = []

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
        
        return generated

    def generate_from_tercet(self, tercet, temperatures, n_verses:int=100):

        '''generates 'n_verses' for each temperature, starting from
        input tercet, where every verse has at most 'tercet_max_len' tokens'''

        # prepare input tercet in order to feed it to the model
        start = list(tf.keras.preprocessing.sequence.pad_sequences(
            [flatten(
                encode_tokens(
                    split_tokens(tercet, self.sep),
                    self.str2idx))],
            maxlen = self.tercet_max_len)[0])

        print("Start:\n", np.array(tercet))        
        print("\nGenerating new cantica: ")

        # generate a cantica for each temperature
        generations = []
        for temp in temperatures:

            print(f"- temperature {temp}... ", end="")

            # start timer
            t_start = time.time()

            # generate cantica
            generated_string = self._generate(start = start,
                                                max_len = self.tercet_max_len,
                                                n_verses = n_verses,
                                                temperature = temp)

            # decode the generated cantica and remove special tokens
            generated_string = clear_text(ints_to_text(generated_string, self.idx2str))

            # append generated cantica to results
            generations.append(generated_string)

            # stop timer
            t_gen = round(time.time() - t_start)
            print(f"completed ({int(t_gen/3600)}h {int(t_gen/60%60)}m {int(t_gen%60)}s)")

        return generations

    ############
    # TRAINING #
    ############        
        
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

    
    def train_model(self, dataset, dataset_name,
                    weights_path:str = "weights/", checkpoint:int = None):

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
            epochs = self.epochs_comedy
            original_length = self.original_length_comedy
        else:
            epochs = self.epochs_production
            original_length = self.original_length_production

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
            if checkpoint:
                if batch != 0 and batch % (original_length*checkpoint) == 0:
                    self.save_model_weights(weights_path)
        
        # append last values to histories
        loss_history.append('{:.4f}'.format(self.train_loss.result()))
        accuracy_history.append('{:.4f}'.format(self.train_accuracy.result()))

        # stop timer
        t = round(time.time() - start)
        print(f'\n\tTraining completed in {int(t/3600)}h {int(t/60%60)}m {int(t%60)}s.\n')
        
        return t, loss_history, accuracy_history

    ###########
    # WEIGHTS #
    ###########

    def get_model_name(self):

        '''stringify the model description for the file name'''

        return "_".join((
            f"{self.encoders}",
            f"{self.decoders}",
            f"{self.heads}",
            f"{self.d_model}",
            f"{self.dff}",
            f"{self.epochs_production}",
            f"{self.epochs_comedy}"
        ))

    def save_model_weights(self, path:str = "weights/"):

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

    def load_model_weights(self, path:str = "weights/"):

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
                    ">> Weights not found for ",
                    f"trained_epochs_production = {self.trained_epochs_production} and ",
                    f"trained_epochs_comedy = {self.trained_epochs_comedy}.",
                    "\nLoading weights from checkpoint with highest number of trained epochs."
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