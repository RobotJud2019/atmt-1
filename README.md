# atmt code base

atmt

### HW3 ###
#### BPE ####

the github repository subword-nmt must be placed in ../snmt

start preprocessing by entering `bash mypreprocess_baseline_data.sh`

start training by entering `python train.py`

or `python train.py --lr LR ` to set the learning rate to *LR*

modifying dropout rate by editing seq2seq/models/lstm.py in the method base_architure

args.encoder_dropout_in = getattr(args, 'encoder_dropout_in', 0.2)

args.encoder_dropout_out = getattr(args, 'encoder_dropout_out', 0.2)

. . .

args.decoder_dropout_in = getattr(args, 'decoder_dropout_in', 0.2)

args.decoder_dropout_out = getattr(args, 'decoder_dropout_out', 0.2)

to set the dropout rate to 0.2

-----------------------------------------------------------------------------------

### HW4 ###

### beam search ###

#### length normalization ####

set alpha in `seq2seq/beam.py`

Equation (14) in the article 

Yonghui Wu et al Google's neural machine translation system: Bridging the gap between human
and machine translation. CoRR, abs/1609.08144, 2016.


#### diverse decoding #### 

set gamma in `seq2seq/beam.py`

select bestN and mutual information for diversity as proposed in section 4.2 in the article : 

Jiwei Li and Daniel Jurafsky. Mutual information and diverse decoding improve neural
machine translation. ArXiv, abs/1601.00372, 2016.

start translating with beam search
`python translate_beam.py --beam-size 5 `       --- for beam-size 5 


