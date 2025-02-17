#!/bin/sh

## setup
dpath=../../data/iwslt14-en-de/raw
tpath=../../data/iwslt14-en-de/tokenizer
vocab_size=10000

## cat both en and de data
cat $dpath/train.en $dpath/val.en $dpath/test.en $dpath/train.de $dpath/val.de $dpath/test.de | shuf > $dpath/train.all.both

## train the vocab
mkdir $tpath
mkdir $tpath/vocab_$vocab_size
python3 vocab_trainer.py --data $dpath/train.all.both --size $vocab_size --output $tpath/vocab_$vocab_size