# atmt code base

atmt

the github repository subword-nmt must be placed in ../snmt

start preprocessing by entering bash mypreprocess_baseline_data.sh

start training by entering python train.py

or python train.py --lr to change the learning rate to

modifying dropout rate by editing seq2seq/models/lstm.py in the method base_architure

args.encoder_dropout_in = getattr(args, 'encoder_dropout_in', 0.2)
args.encoder_dropout_out = getattr(args, 'encoder_dropout_out', 0.2)

. . .

args.decoder_dropout_in = getattr(args, 'decoder_dropout_in', 0.2)
args.decoder_dropout_out = getattr(args, 'decoder_dropout_out', 0.2)

to set the dropout rate to 0.2
