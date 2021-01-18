#!/bin/sh
dir="text_clsf_lib/preprocessing/vectorization/resources/embeddings/glove/twitter"
mkdir -p $dir && \
cd $dir && \
curl http://downloads.cs.stanford.edu/nlp/data/wordvecs/glove.twitter.27B.zip > glove_twitter.zip && \
unzip glove_twitter.zip && \
rm glove_twitter.zip
mv glove.twitter.27B.25d.txt 25d.txt
mv glove.twitter.27B.50d.txt 50d.txt
mv glove.twitter.27B.100d.txt 100d.txt
mv glove.twitter.27B.200d.txt 200d.txt

