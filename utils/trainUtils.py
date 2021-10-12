import time

#############
# OPTIMIZER #
#############

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()
    
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps
    
  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
  
learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)

####################
# LOSS AND METRICS #
####################

# Sparse Categorical Crossentropy loss function
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
  # "mask" is a boolean tensor with False values on padding values (0 values) 
  mask = tf.math.logical_not(tf.math.equal(real, 0))2
  # "loss_" is a tensor of float values
  loss_ = loss_object(real, pred)
  # convert mask boolean values to float (False=0. and True=1.)
  mask = tf.cast(mask, dtype=loss_.dtype)
  # apply mask to loss tensor
  loss_ *= mask
  
  # returns a single float value representing the loss value
  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

# training metrics definition
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')

#########################
# TRAINING PROGRESS BAR #
#########################

# prints model training progress
def print_progress(done:int, total:int, *args):
  maxlen = 25
  bars = round(done*maxlen/total)
  print("\r[{}{}] {:3}%".format("="*int(bars),
                                " "*int((maxlen - bars)),
                                round(done*100/total)), 
        end="\t {:>5}/{:<5}\t{}\t".format(done,
                                          total,
                                          [str(a) for a in args]))

######################
# TRAINING FUNCTIONS #
######################
 
train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
]

@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
 
  pred_size = 1
  # split input and target
  tar_inp = tar[:, :-pred_size]
  tar_real = tar[:, pred_size:]
 
  # create masks
  enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
  
  # compute predictions
  with tf.GradientTape() as tape:
    predictions, _ = transformer(inp,
                                 tar_inp, 
                                 True, 
                                 enc_padding_mask,
                                 combined_mask,
                                 dec_padding_mask
                                 )
 
    # compute loss function
    loss = loss_function(tar_real, predictions)
  
  # compute gradients
  gradients = tape.gradient(loss, transformer.trainable_variables)    
 
  # apply gradients
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
 
  train_loss(loss)
  train_accuracy(tar_real, predictions)

 
def train_model(dataset, epochs, real_size):

  start = time.time()
  
  repetition = 1
  
  loss_history = []
  accuracy_history = []

  for epoch in range(epochs):
    
    repetitions = int(len(dataset)/real_size)

    for (batch, (inp, tar)) in enumerate(dataset):
            
      train_step(inp, tar)
 
      print_progress(batch, len(dataset), 
                    "epoch {}/{}   repetition {}/{}  loss: {:.4f}  accuracy: {:.4f}".format(
                        epoch+1, epochs, repetition, repetitions, train_loss.result(), train_accuracy.result()))

      # at each repetition "epoch"
      if batch != 0 and (batch) % real_size == 0:
        # Append values to histories
        loss_history.append('{:.4f}'.format(train_loss.result()))
        accuracy_history.append('{:.4f}'.format(train_accuracy.result()))
        # Reset loss and accuracy states
        train_loss.reset_states()
        train_accuracy.reset_states()
        repetition +=1
  
  # Append last values to histories
  loss_history.append('{:.4f}'.format(train_loss.result()))
  accuracy_history.append('{:.4f}'.format(train_accuracy.result()))

  t = round(time.time() - start)
  print(f'\n\tTraining completed in {int(t/3600)}h {int(t/60%60)}m {int(t%60)}s.\n')
  
  return t, loss_history, accuracy_history

####### CALL
t_comedy, loss_hist_comedy, acc_hist_comedy = train_model(dataset_comedy, epochs_comedy, real_size_comedy)
