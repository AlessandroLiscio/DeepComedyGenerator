import time
import tensorflow as tf
from transformer import Transformer, create_masks
from utils.training import print_progress, CustomSchedule

_train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
]

class Generator():

    def __init__(self, vocab_size:int, encoders:int = 5, decoders:int = 5, heads:int = 4, d_model:int = 256, dff:int = 512, dropout:float = 0.2):
    
        # initialize transformer model parameters
        self.vocab_size = vocab_size
        self.encoders = encoders
        self.decoders = decoders
        self.heads = heads
        self.d_model = d_model
        self.dff = dff
        self.dropout = dropout

        # transformer model instantiation
        self.transformer = Transformer(encoders,
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
        return f"""
Generator parameters:
- encoders: {self.encoders}
- decoders: {self.decoders}
- num_heads: {self.heads}
- d_model: {self.d_model}
- dff: {self.dff}
- dropout: {self.dropout}
- optimizer: {str(type(self.optimizer))[:-2].split('.')[-1]}
- loss: {str(type(self.loss_object))[:-2].split('.')[-1]}
- metric: {str(type(self.train_accuracy))[:-2].split('.')[-1]}
"""

    ##############
    # GENERATION #
    ##############

    def evaluate(self, input, eov, max_length=100, temperature=1.0):
    
        # add the batch dimension for compatibility
        encoder_input = tf.expand_dims(input, 0)
        decoder_input = tf.expand_dims(input, 0)
        
        # the final output of the evaluation (initially, this is an empty list)
        output = []
        
        # we repeat the process to get the entire verse (end-of-verse token is predicted)
        for i in range(max_length):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, decoder_input)  
            logits, attention_weights = self.transformer(
                encoder_input, decoder_input, False,
                enc_padding_mask, combined_mask, dec_padding_mask
            )
        
            # the higher the temperature, the more original (or crazy) is the text
            predictions = logits[: ,:, :]
            predictions /= temperature
            predicted_id = tf.cast(tf.random.categorical(tf.squeeze(predictions, 0), num_samples=1)[-1,0].numpy() , tf.int32)
            
            # append the predicted token to the output
            output.append(predicted_id)
        
            # if the token coincides with the end-of-verse token
            if predicted_id == eov: break
        
            # otherwise the token is appended both to the new decoder input
            decoder_input = tf.concat([decoder_input, [[predicted_id]]], axis=-1)
        
        return output, attention_weights

    def generate(self, str2idx, start, eov, max_len, max_iterations=5, temperature=1.0):

        # drop the first verse to keep a window of 3 verses
        def drop_first_verse(sequence):
            for i, element in enumerate(sequence):
                if element == str2idx['</v>']:
                    return sequence[i+1:]

        # variables initialization
        input_sequence = start.copy()
        generated = []

        for i in range(max_iterations):

            # pad the input list to reach the max_len
            input_list = list(tf.keras.preprocessing.sequence.pad_sequences([input_sequence], maxlen=max_len)[0])

            # generate one verse
            generated_verse, _ = self.evaluate(input_list, eov, temperature=temperature)

            # append the generated verse to the input sequence
            input_sequence += generated_verse
            # drop the first verse to keep a window of 3 verses
            input_sequence = drop_first_verse(input_sequence)

            # append the generated verse to the output
            generated += generated_verse
        
        return generated

    ############
    # TRAINING #
    ############

    def _loss_function(self, real, pred):
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
            
        # split input and target
        pred_size = 1
        tar_inp = tar[:, :-pred_size]
        tar_real = tar[:, pred_size:]
        
        # create masks
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
        
        # compute predictions
        with tf.GradientTape() as tape:
            predictions, _ = self.transformer(inp,
                                        tar_inp, 
                                        True, 
                                        enc_padding_mask,
                                        combined_mask,
                                        dec_padding_mask
                                        )
        
            # compute loss function
            loss = self._loss_function(tar_real, predictions)
        
        # compute gradients
        gradients = tape.gradient(loss, self.transformer.trainable_variables)    
        
        # apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))
        
        # update training metrics
        self.train_loss(loss)
        self.train_accuracy(tar_real, predictions)

    
    def train_model(self, dataset, epochs, real_size):

        # start timer
        start = time.time()
        
        # initialize training variables
        repetition = 1
        loss_history = []
        accuracy_history = []

        for epoch in range(epochs):
            
            # compute total repetitions needed
            repetitions = int(len(dataset)/real_size)

            for (batch, (inp, tar)) in enumerate(dataset):
                    
                # update gradients
                self._train_step(inp, tar)
            
                # show/update output progress bar
                print_progress(batch, len(dataset), 
                                "epoch {}/{}   repetition {}/{}  loss: {:.4f}  accuracy: {:.4f}".format(
                                    epoch+1, epochs, repetition, repetitions, self.train_loss.result(), self.train_accuracy.result()))

                # update metrics and training history at each repetition ("epoch")
                if batch != 0 and (batch) % real_size == 0:

                    # Append values to histories
                    loss_history.append('{:.4f}'.format(self.train_loss.result()))
                    accuracy_history.append('{:.4f}'.format(self.train_accuracy.result()))

                    # Reset loss and accuracy states
                    self.train_loss.reset_states()
                    self.train_accuracy.reset_states()
                    repetition +=1
        
        # append last values to histories
        loss_history.append('{:.4f}'.format(self.train_loss.result()))
        accuracy_history.append('{:.4f}'.format(self.train_accuracy.result()))

        # stop timer
        t = round(time.time() - start)
        print(f'\n\tTraining completed in {int(t/3600)}h {int(t/60%60)}m {int(t%60)}s.\n')
        
        return t, loss_history, accuracy_history