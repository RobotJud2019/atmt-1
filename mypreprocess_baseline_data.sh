cat baseline/raw_data/train.de | perl moses_scripts/normalize-punctuation.perl -l de | perl moses_scripts/tokenizer.perl -l de -a -q > baseline/preprocessed_data/train.de.p

cat baseline/raw_data/train.en | perl moses_scripts/normalize-punctuation.perl -l en | perl moses_scripts/tokenizer.perl -l en -a -q > baseline/preprocessed_data/train.en.p

perl moses_scripts/train-truecaser.perl --model baseline/preprocessed_data/tm.de --corpus baseline/preprocessed_data/train.de.p

perl moses_scripts/train-truecaser.perl --model baseline/preprocessed_data/tm.en --corpus baseline/preprocessed_data/train.en.p

cat baseline/preprocessed_data/train.de.p | perl moses_scripts/truecase.perl --model baseline/preprocessed_data/tm.de > baseline/preprocessed_data/train.de 

cat baseline/preprocessed_data/train.en.p | perl moses_scripts/truecase.perl --model baseline/preprocessed_data/tm.en > baseline/preprocessed_data/train.en

cat baseline/raw_data/valid.de | perl moses_scripts/normalize-punctuation.perl -l de | perl moses_scripts/tokenizer.perl -l de -a -q | perl moses_scripts/truecase.perl --model baseline/preprocessed_data/tm.de > baseline/preprocessed_data/valid.de

cat baseline/raw_data/valid.en | perl moses_scripts/normalize-punctuation.perl -l en | perl moses_scripts/tokenizer.perl -l en -a -q | perl moses_scripts/truecase.perl --model baseline/preprocessed_data/tm.en > baseline/preprocessed_data/valid.en

cat baseline/raw_data/test.de | perl moses_scripts/normalize-punctuation.perl -l de | perl moses_scripts/tokenizer.perl -l de -a -q | perl moses_scripts/truecase.perl --model baseline/preprocessed_data/tm.de > baseline/preprocessed_data/test.de

cat baseline/raw_data/test.en | perl moses_scripts/normalize-punctuation.perl -l en | perl moses_scripts/tokenizer.perl -l en -a -q | perl moses_scripts/truecase.perl --model baseline/preprocessed_data/tm.en > baseline/preprocessed_data/test.en

cd ../snmt
python subword_nmt/learn_bpe.py -s  4000 < ../atmt/baseline/preprocessed_data/train.de > ../atmt/baseline/preprocessed_data/bpecodes_train.de
python subword_nmt/learn_bpe.py -s  4000 < ../atmt/baseline/preprocessed_data/train.en > ../atmt/baseline/preprocessed_data/bpecodes_train.en
python subword_nmt/learn_bpe.py -s  4000 < ../atmt/baseline/preprocessed_data/valid.de > ../atmt/baseline/preprocessed_data/bpecodes_valid.de
python subword_nmt/learn_bpe.py -s  4000 < ../atmt/baseline/preprocessed_data/valid.en > ../atmt/baseline/preprocessed_data/bpecodes_valid.en
python subword_nmt/learn_bpe.py -s  4000 < ../atmt/baseline/preprocessed_data/test.de > ../atmt/baseline/preprocessed_data/bpecodes_test.de
python subword_nmt/learn_bpe.py -s  4000 < ../atmt/baseline/preprocessed_data/test.en > ../atmt/baseline/preprocessed_data/bpecodes_test.en


python subword_nmt/apply_bpe.py -c ../atmt/baseline/preprocessed_data/bpecodes_train.de  < ../atmt/baseline/preprocessed_data/train.de > ../atmt/baseline/preprocessed_data/bpe/train.de
python subword_nmt/apply_bpe.py -c ../atmt/baseline/preprocessed_data/bpecodes_train.en  < ../atmt/baseline/preprocessed_data/train.en > ../atmt/baseline/preprocessed_data/bpe/train.en
python subword_nmt/apply_bpe.py -c ../atmt/baseline/preprocessed_data/bpecodes_valid.de  < ../atmt/baseline/preprocessed_data/valid.de > ../atmt/baseline/preprocessed_data/bpe/valid.de
python subword_nmt/apply_bpe.py -c ../atmt/baseline/preprocessed_data/bpecodes_valid.en  < ../atmt/baseline/preprocessed_data/valid.en > ../atmt/baseline/preprocessed_data/bpe/valid.en
python subword_nmt/apply_bpe.py -c ../atmt/baseline/preprocessed_data/bpecodes_test.de  < ../atmt/baseline/preprocessed_data/test.de > ../atmt/baseline/preprocessed_data/bpe/test.de
python subword_nmt/apply_bpe.py -c ../atmt/baseline/preprocessed_data/bpecodes_test.en  < ../atmt/baseline/preprocessed_data/test.en > ../atmt/baseline/preprocessed_data/bpe/test.en

cd ../atmt

cd ./baseline/preprocessed_data

rm train.de train.en valid.de valid.en test.de test.en

cd ../..

rm baseline/preprocessed_data/train.de.p
rm baseline/preprocessed_data/train.en.p

python preprocess.py --target-lang en --source-lang de --dest-dir baseline/prepared_data/ --train-prefix baseline/preprocessed_data/bpe/train --valid-prefix baseline/preprocessed_data/bpe/valid --test-prefix baseline/preprocessed_data/bpe/test --threshold-src 1 --threshold-tgt 1 --num-words-src 4000 --num-words-tgt 4000
