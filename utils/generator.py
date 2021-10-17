import time
import tensorflow as tf
from utils.transformer import Transformer, create_masks
from utils.training import print_progress, CustomSchedule
from utils.preprocessing import *

_train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
]

class Generator():

    def __init__(self, vocab_size:int, str2idx, idx2str, encoders:int = 5, decoders:int = 5, heads:int = 4, d_model:int = 256, dff:int = 512, dropout:float = 0.2):
    
        # initialize transformer model parameters
        self.vocab_size = vocab_size
        self.str2idx = str2idx
        self.idx2str = idx2str
        self.encoders = encoders
        self.decoders = decoders
        self.heads = heads
        self.d_model = d_model
        self.dff = dff
        self.dropout = dropout

        # transformer model instantiation
        self.model = Transformer(encoders,
                                decoders,
                                d_model,
                                heads,
                                dff,
                                input_vocab_size= vocab_size,
                                target_vocab_size= vocab_size,
                                pe_input= vocab_size, 
                                pe_target= vocab_size,
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
            "Generator parameters:",
            f"- encoders: {self.encoders}",
            f"- decoders: {self.decoders}",
            f"- num_heads: {self.heads}",
            f"- d_model: {self.d_model}",
            f"- dff: {self.dff}",
            f"- vocab_size: {self.vocab_size}",
            f"- dropout: {self.dropout}",
            f"- optimizer: {str(type(self.optimizer))[:-2].split('.')[-1]}",
            f"- loss: {str(type(self.loss_object))[:-2].split('.')[-1]}",
            f"- metric: {str(type(self.train_accuracy))[:-2].split('.')[-1]}",
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
            for i, element in enumerate(sequence):
                if element == eov:
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
                                                    max_len = max_len,
                                                    temperature=temperature)

            # append the generated verse to the input sequence
            input_sequence += generated_verse

            # drop the first verse to keep a window of 3 verses
            input_sequence = drop_first_verse(input_sequence)

            # append the generated verse to the output
            generated += generated_verse
        
        return generated

    def generate_from_tercet(self, tercet, temperatures, max_len:int, n_verses:int=100):

        '''generates 'n_verses' for each temperature, starting from
        input tercet, where every verse has at most 'max_len' tokens'''

        # prepare input tercet in order to feed it to the model
        start = list(tf.keras.preprocessing.sequence.pad_sequences(
            [flatten(
                encode_tokens(
                    split_tokens(tercet),
                    self.str2idx))],
            maxlen = max_len)[0])

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
                                                max_len = max_len,
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

    
    def train_model(self, dataset, original_length:int):

        '''train model on target dataset. As the dataset has been
        repeated several times (each time singularly shuffled)
        instead of using the concept of "epochs" on the same dataset,
        the original dataset length is required as input, in order to
        update training metrics and histories at each dataset
        repetition (which basically counts as an "epoch")'''

        # start timer
        start = time.time()
        
        # initialize training variables
        epoch = 1
        loss_history = []
        accuracy_history = []
            
        # compute original dataset size
        epochs = int(len(dataset)/original_length)

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
            if batch != 0 and (batch) % original_length == 0:

                # Append values to histories
                loss_history.append('{:.4f}'.format(self.train_loss.result()))
                accuracy_history.append('{:.4f}'.format(self.train_accuracy.result()))

                # Reset loss and accuracy states
                self.train_loss.reset_states()
                self.train_accuracy.reset_states()
                epoch +=1
        
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

    def save_weights(self, epoch:int, out_path:str = "results/"):

        '''saves the weights of the model to target path. The name
        of the file is based on the instantiated model's parameters and,
        if given, number of training epochs'''

        # stringify the model description for the file name
        model_description = "_".join((
            f"{self.model.encoders}",
            f"{self.model.decoders}",
            f"{self.model.heads}",
            f"{self.model.d_model}",
            f"{self.model.dff}",
            f"{str(epoch)}"
        ))

        # create weights folder if it doesn't exist
        w_path = out_path + "weights/"
        create_folder(w_path)

        # save weights
        try:
            w_path = w_path+model_description
            self.model.save_weights(w_path)
            print(f"Generator model weights saved to: {w_path}" )
        except e:
            print(f"ERROR: problem saving weights to {w_path}")

    def load_weights(self, epoch:int, path:str = "weights/"):

        '''loads the weights of the model from input path, based on the
        instantiated model's parameters and, if given, number of
        training epochs'''

        # stringify the model description for the file name
        model_description = "_".join((
            f"{self.model.encoders}",
            f"{self.model.decoders}",
            f"{self.model.d_model}",
            f"{self.model.dff}",
            f"{self.model.heads}",
            f"{str(epoch)}" 
        ))

        # load weights
        try:
            w_path = path+model_description
            self.model.load_weights(w_path)
            print(f"Generator model weights loaded from: {w_path}" )
        except e:
            print(f"ERROR: problem loading weights from {w_path}")