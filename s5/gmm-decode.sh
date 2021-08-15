. ./path-gmm.sh
. parse_options.sh

train_cmd=run.pl
nj=1

rm -rf transcriptions/subsegments transcriptions/utt2num_frames transcriptions/utt2dur

ffmpeg -y -i $1 -acodec pcm_s16le -ac 1 -ar 16000 wav/output.wav

# AUDIO --> FEATURE VECTORS
compute-mfcc-feats \
    --config=conf/mfcc.conf \
    scp:transcriptions/wav.scp \
    ark,scp:transcriptions/feats.ark,transcriptions/feats.scp

# Create vad
steps/compute_vad_decision.sh --nj $nj --vad-config conf/vad.conf --cmd "$train_cmd" \
    transcriptions transcriptions/make_vad transcriptions/mfcc

# Create segments
. ./vad_to_segment.sh --nj $nj --cmd "$train_cmd" \
    transcriptions transcriptions/segmented

awk '{print $1 " " $2}' transcriptions/segmented/segments > transcriptions/segmented/utt2spk
utils/utt2spk_to_spk2utt.pl transcriptions/segmented/utt2spk > transcriptions/segmented/spk2utt

# COMPUTE CMVN
compute-cmvn-stats \
    --spk2utt=ark:transcriptions/segmented/spk2utt \
    scp:transcriptions/segmented/feats.scp \
    ark,scp:transcriptions/cmvn.ark,transcriptions/cmvn.scp

# FEATURE VECTORS + CMVN --> LATTICE
gmm-latgen-faster \
    --max-active=7000 --beam=13.0 --lattice-beam=6.0 --acoustic-scale=0.083333 --allow-partial=true \
    --word-symbol-table=./exp/tri2b/graph/words.txt ./exp/tri2b/final.mdl ./exp/tri2b/graph/HCLG.fst \
    'ark,s,cs:apply-cmvn --utt2spk=ark:transcriptions/segmented/utt2spk scp:transcriptions/cmvn.scp scp:transcriptions/segmented/feats.scp ark:- | splice-feats --left-context=3 --right-context=3 ark:- ark:- | transform-feats ./exp/tri2b/final.mat ark:- ark:- |' 'ark:transcriptions/lattices.ark' 

lattice-to-ctm-conf \
  --acoustic-scale=0.08333 \
  ark:transcriptions/lattices.ark transcriptions/pb.ctm

int2sym.pl -f 5-5 exp/tri2b/graph/words.txt transcriptions/pb.ctm > transcriptions/text

# LATTICE --> BEST PATH THROUGH LATTICE
lattice-best-path \
    --word-symbol-table=exp/tri2b/graph/words.txt \
    --acoustic-scale=0.083333 \
    ark:transcriptions/lattices.ark \
    ark,t:transcriptions/one-best.tra

# BEST PATH INTERGERS --> BEST PATH WORDS
utils/int2sym.pl -f 2- \
    exp/tri2b/graph/words.txt \
    transcriptions/one-best.tra \
    > transcriptions/one-best.txt

awk '{for (i=2; i<NF; i++) printf $i " "; print $NF}' transcriptions/one-best.txt > transcriptions/one-best-text.txt
