. ./path-gmm.sh

online-audio-server-decode-faster --verbose=1 --rt-min=0.5 --rt-max=3.0 --max-active=7000 \
--beam=13.0 --acoustic-scale=0.083333 --left-context=3 --right-context=3 \
exp/tri2b/final.mdl exp/tri2b/graph/HCLG.fst exp/tri2b/graph/words.txt '1:2:3:4:5' \
data/lang/phones/word_boundary.int 5010 exp/tri2b/final.mat
