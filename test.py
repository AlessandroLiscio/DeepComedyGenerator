#############
# GENERATOR #
#############

from generator import Generator
generator = Generator(vocab_size = 50,
                        encoders = 5, 
                        decoders = 5, 
                        d_model = 256,
                        dff = 512,
                        heads = 4,
                        dropout = 0.2)

print(generator)