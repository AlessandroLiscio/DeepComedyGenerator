# DeepComedyGenerator
## USAGE

It is possible to run this repository in three different ways:
 - colab
 - local
 - slurm

The paths for the input and output folder are automatically assigned inside the "parser.py" script
based on the value of the 'runtime' variable.

### FROM COLAB

In order to run the code from a Colab session, follow these steps:

 1- Open this notebook: https://colab.research.google.com/drive/1l_PAqCU6UaZsx4Z8SK9FNMswP5yv7K1O#scrollTo=YLZyIDgbgZHe
 2- In the colab session, create a directory named "src" and upload the script in the repository "src" directory in it
 3- In your GDrive account, create a directory named "DC-gen" upload the repository "data" director in it
 4- (Optional) Check the "runtime" variable is set to 'colab'
 5- (Optional) Change the values in the "Parser" call

### FROM TERMINAL

To change the default behaviour, just add as command parameters
any of the following listed parameters as "--arg $arg_value".

N.B.: For the boolean arguments to be set as True, there is no need to
specify the value, just add the command parameter as "--arg".

An example could be the following:
```
python main.py --dataset 'sov_sot' --dropout 0.2 --train --generate
```

## DEFAULT SETUP
#### RUN INFO
from_pretrained=False
train=False
generate=False
#### DATASET INFO
dataset='sov_sot'
comedy_name='comedy_np'
tokenization='base'
#### DATASET PROCESSING
stop=['</v>']
padding='pre'
inp_len=3
tar_len=4
#### MODEL PARAMETERS
encoders=5
decoders=5
heads=4
d_model=256
dff=512
dropout=0.2
#### TRAINING INFO
epochs_production=0
epochs_comedy=150
checkpoint=10
weight_eov=1.0
weight_sot=1.0
#### VEROBSE
verbose=True
