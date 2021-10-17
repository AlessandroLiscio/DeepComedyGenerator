import tensorflow as tf

#############
# OPTIMIZER #
#############

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

#########################
# TRAINING PROGRESS BAR #
#########################

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