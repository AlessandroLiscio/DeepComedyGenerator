# DeepComedyGenerator: Generating cantos in Divine Comedy-style using transformers

The artificial generation of poetries has been proposed several times as a way
to demonstrate the performance of sequence-to-sequence models such as RNNs and
LSTMs used in transfer learning tasks oriented to capture the intrinsic characteristics of a
set of textual documents and producing a novel text with the same style. The standard
solutions proposed so far copied the writing styles and the structure of the text but in some
cases it is too difficult to catch some more hidden and complex characteristics, like rhymes,
and keep the references to the context for a long period. With this work we want to
demonstrate how good transformers are in performing those tasks, even on the relatively
small dataset of the Divine Comedy, by reproducing the Dante’s style of writing and
vocabulary, in some case imitating the Tuscan involved language, by building new
nonexistent but convincing words in a sort of metasemantic poems, perfectly respecting the
rules of terza rima and hendecasyllables, without any type of external heuristic
intervention.

# A quick demo of our results:

  
  *E io vidi con le genti in novelle*
  
  *più alto poco dir per la vista corte*
  
  *rivolge a la cagion che non favelle.*

  *ma dimmi quei che pace a te forte,*
  
  *se troppa luce, che la gente nota,*
  
  *rimembrar dal fatto hai mano inforte!.*

  *ed ella: io di onde qui rinota,*
  
  *rispuos'io lui, ti giova prìa ch'i' tolsi*
  
  *come mano a costui fu sì commota.*
  
## USAGE

It is possible to run this repository in three different ways:
 - colab
 - local
 - slurm

The paths for the input and output folder are automatically assigned inside the "parser.py" script
based on the value of the 'runtime' variable.

### FROM COLAB

In order to run the code from a Colab session, follow these steps:

 1. Open this notebook: https://colab.research.google.com/drive/1l_PAqCU6UaZsx4Z8SK9FNMswP5yv7K1O#scrollTo=YLZyIDgbgZHe
 2. In the colab session, create a directory named "src" and upload the script in the repository "src" directory in it
 3. In your GDrive account, create a directory named "DC-gen" upload the repository "data" director in it
 4. (Optional) Check the "runtime" variable is set to 'colab'
 5. (Optional) Change the values in the "Parser" call

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
 - from_pretrained   = False
 - train             = False
 - generate          = False
#### DATASET INFO
 - dataset           = 'sov'
 - comedy_name       = 'comedy_np'
 - tokenization      = 'es'
#### DATASET PROCESSING
 - stop              = \['\</v>']
 - padding           = 'pre'
 - inp_len           = 3
 - tar_len           = 4
#### MODEL PARAMETERS
 - encoders          = 5
 - decoders          = 5
 - heads             = 4
 - d_model           = 256
 - dff               = 512
 - dropout           = 0.2
#### TRAINING INFO
 - epochs_production = 0
 - epochs_comedy     = 70
 - checkpoint        = 10
 - weight_eov        = 1.0
 - weight_sot        = 1.0
#### VERBOSITY
 - verbose           = True
