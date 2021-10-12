import tensorflow as tf
from transformer import Transformer

class Generator():

    # __slots__ = [vocab_size, encoders, decoders, d_model, dff, heads, dropout, transformer]

    def __init__(self, vocab_size:int , encoders:int = 5, decoders:int = 5, d_model:int = 256, dff:int = 512, heads:int = 4, dropout:float = 0.2):
    
        self.vocab_size = vocab_size
        self.encoders = encoders
        self.decoders = decoders
        self.d_model = d_model
        self.dff = dff
        self.heads = heads
        self.dropout = dropout

        # trainsformer model instantiation
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

    def __str__(self):
        return f"""
Generator parameters:
- encoders: {self.encoders}
- decoders: {self.decoders}
- d_model: {self.d_model}
- dff: {self.dff}
- num_heads: {self.heads}
- dropout: {self.dropout}
"""

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
            if predicted_id == eov:
                break
        
            # otherwise the token is appended both to the new decoder input
            decoder_input = tf.concat([decoder_input, [[predicted_id]]], axis=-1)
        
        return output, attention_weights

    def generate(self, start, eov, max_len, max_iterations=5, temperature=1.0):

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
            generated_verse, _ = evaluate(input_list, eov, temperature=temperature)

            # append the generated verse to the input sequence
            input_sequence += generated_verse
            # drop the first verse to keep a window of 3 verses
            input_sequence = drop_first_verse(input_sequence)

            # append the generated verse to the output
            generated += generated_verse
        
        return generated
