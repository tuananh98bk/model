#!/usr/bin/env bash

date --date "7 hour"

. ./cmd.sh
. ./path.sh # Needed for KALDI_ROOT
. ./config.sh # Needed for REC_ROOT and WAV_ROOT

# Check wave file directory
if [ ! -d $WAV_ROOT ]; then
  echo "Cannot find wav directory $WAV_ROOT"
  echo "Please set the WAV_ROOT"
  exit 1;
fi

# Define stage (useful for skipping some stagess)
stage=0
. parse_options.sh || exit 1;

# Define number of parallel jobs
njobs=$(nproc)

# Silence boost factor
boost_silence=1.00

# Setup feature file directory
mfcc="$REC_ROOT/mfcc"
mkdir -p $mfcc

# Setup log file directory
exp="$REC_ROOT/exp"
mkdir -p $exp

# Setup other relevant directories
data="$REC_ROOT/data"
lang="$data/lang"
dict="$data/local/dict"
langtmp="$data/local/lang"
mkdir -p $langtmp
steps="steps"
utils="utils"
tot_gauss_mono=1000
num_leaves_tri=1000
tot_gauss_tri=2000
num_iters_mono=25
num_iters_tri=25

# Data preparation
#if [ $stage -le 0 ]; then
#  echo ""
#  echo "Stage 0: Preparing data"
#  rm -rf $data/*
#  local/vn_prepare_data.sh || exit 1
#fi


# Language model preparation
if [ $stage -le 1 ]; then
  echo ""
  echo "Stage 1: Preparing lang"
  rm -rf $lang
  rm -rf $langtmp
  local/vn_prepare_dict.sh || exit 1
  $utils/prepare_lang.sh --num-sil-states 5 \
     --num-nonsil-states 3 \
     --position-dependent-phones false \
     --share-silence-phones true \
     $dict "<UNK>" $langtmp $lang || exit 1
  local/vn_prepare_grammar.sh || exit 1
fi


# Feature extraction
set_list="train test"                                                                                                           
if [ $stage -le 2 ]; then
  echo ""
  echo "Stage 2: Extracting mfcc features"
  # rm -rf $mfcc/*

  for x in $set_list; do 
    if [ -d $data/$x ]; then
      $steps/make_mfcc.sh --nj $njobs --cmd "$train_cmd" $data/$x $exp/make_mfcc/$x $mfcc || exit 1
      $steps/compute_cmvn_stats.sh $data/$x $exp/make_mfcc/$x $mfcc || exit 1
    fi
  done
fi

# Training
if [ $stage -le 3 ]; then
  echo ""
  echo "Stage 3: Starting training"
  rm -rf $exp/*

  $steps/train_mono.sh --nj $njobs --cmd "$train_cmd" \
    --boost_silence $boost_silence \
    $data/train $lang $exp/mono0a || exit 1;

  $steps/align_si.sh --nj $njobs --cmd "$train_cmd" \
    --boost_silence $boost_silence \
    $data/train $lang $exp/mono0a $exp/mono0a_ali || exit 1;

  $steps/train_deltas.sh --cmd "$train_cmd" \
    --boost_silence $boost_silence \
    2000 10000 $data/train $lang $exp/mono0a_ali $exp/tri1 || exit 1;

  $steps/align_si.sh --nj $njobs --cmd "$train_cmd" \
    $data/train $lang $exp/tri1 $exp/tri1_ali || exit 1;

  # $steps/train_deltas.sh --cmd "$train_cmd" \
  #   2500 15000 $data/train $lang $exp/tri1_ali $exp/tri1b || exit 1;

  # $steps/align_si.sh  --nj $njobs --cmd "$train_cmd" \
  #   --use-graphs true $data/train $lang $exp/tri1b $exp/tri1b_ali  || exit 1;

  $steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" \
    2500 25000 $data/train $lang $exp/tri1_ali $exp/tri2b || exit 1;

  $steps/align_fmllr.sh --nj $njobs --cmd $train_cmd \
    $data/train $lang $exp/tri2b $exp/tri2b_ali || exit 1; 

  $steps/train_sat.sh \
    --cmd $train_cmd \
    3500 40000 $data/train $lang $exp/tri2b_ali exp/tri3 || exit 1;

  $steps/align_fmllr.sh  --cmd "$train_cmd" \
    $data/train $lang $exp/tri3 $exp/tri3_ali || exit 1;

  # dnn_mem_reqs="mem_free=1.0G,ram_free=0.2G" # Lệnh thiết lập khởi tạo vùng nhớ.
  # dnn_extra_opts="--num_epochs 20 --num-epochs-extra 10 --add-layers-period 1 --shrink-interval 3"
  # $steps/nnet2/train_tanh.sh --mix-up 5000 --initial-learning-rate 0.015 \
  #   --final-learning-rate 0.002 --num-hidden-layers 6 \
  #   --num-jobs-nnet $njobs --cmd "$train_cmd" "${dnn_train_extra_opts[@]}" \
  # $data/train $lang $exp/tri3_ali $exp/tri4

  $utils/mkgraph.sh $lang $exp/tri2b $exp/tri2b/graph || exit 1;
  $utils/mkgraph.sh $lang $exp/tri3 $exp/tri3/graph || exit 1;
fi
  
Decoding
if [ $stage -le 4 ]; then
set_list="test"
  echo ""
  echo "Stage 4: Starting decoding"
  rm -rf $exp/*/decode*

  for x in $set_list; do
    if [ -d "$data/$x" ]; then
      $steps/decode.sh --config conf/decode.config --nj $njobs --cmd "$decode_cmd" \
        $exp/tri2b/graph $data/$x $exp/tri2b/decode_$x || exit 1;
      $steps/decode.sh --config conf/decode.config --nj $njobs --cmd "$decode_cmd" \
        $exp/tri3/graph $data/$x $exp/tri3/decode_$x || exit 1;
      # [ ! -d $exp/tri4/decode_$x ] && mkdir -p $exp/tri4/decode_$x 
      # decode_extra_opts=(--num-threads 6) 
      # $steps/nnet2/decode.sh --cmd "$decode_cmd" --nj $njobs "${decode_extra_opts[@]}" \
      #   --max-active 250 \
      #   --min-active 100 \
      #   $exp/tri3/graph $data/$x $exp/tri4/decode_$x | tee $exp/tri4/decode_$x/decode.log
    fi
  done
fi

# Setup nnet3
# if [ $stage -le 5 ]; then 

#     echo "### ============================ ###";
#     echo "### CREATE CONFIG FILES FOR NNET ###";
#     echo "### ============================ ###";

#     mkdir -p $exp/nnet3/easy/configs

#     feat_dim=`feat-to-dim scp:$data/train/feats.scp -`
#     num_targets=`tree-info $exp/tri3_ali/tree 2>/dev/null | grep num-pdfs | awk '{print $2}'` || exit 1;
#     hidden_dim=8

#     cat <<EOF > $exp/nnet3/easy/configs/network.xconfig
#     input dim=$feat_dim name=input
#     relu-renorm-layer name=tdnn1 input=Append(input@-2,input@-1,input,input@1,input@2) dim=$hidden_dim
#     relu-renorm-layer name=tdnn2 dim=$hidden_dim
#     relu-renorm-layer name=tdnn3 input=Append(-1,2) dim=$hidden_dim
#     relu-renorm-layer name=tdnn4 input=Append(-3,3) dim=$hidden_dim
#     relu-renorm-layer name=tdnn5 input=Append(-3,3) dim=$hidden_dim
#     relu-renorm-layer name=tdnn6 input=Append(-3,3) dim=$hidden_dim
#     relu-renorm-layer name=tdnn7 input=Append(-3,3) dim=$hidden_dim
#     relu-renorm-layer name=tdnn8 input=Append(-3,3) dim=$hidden_dim
#     relu-renorm-layer name=tdnn9 input=Append(-3,3) dim=$hidden_dim
#     relu-renorm-layer name=tdnn10 input=Append(-3,3) dim=$hidden_dim
#     relu-renorm-layer name=tdnn11 input=Append(-3,3) dim=$hidden_dim
#     relu-renorm-layer name=tdnnFINAL input=Append(-3,3) dim=$hidden_dim
#     relu-renorm-layer name=prefinal-affine-layer input=tdnnFINAL dim=$hidden_dim
#     output-layer name=output dim=$num_targets max-change=1.5	 
# EOF

#     $steps/nnet3/xconfig_to_configs.py \
#         --xconfig-file $exp/nnet3/easy/configs/network.xconfig \
#         --config-dir $exp/nnet3/easy/configs/
# fi

# if [ $stage -le 6 ]; then 
#   echo "### =================== ###"
#   echo "### MAKE NNET3 EGS DIR  ###"
#   echo "### =================== ###"

#   $steps/nnet3/get_egs.sh \
# 	--cmd "$train_cmd" \
# 	--cmvn-opts "--norm-means=false --norm-vars=false" \
#         --left-context 30 \
#         --right-context 31 \
# 	$data/train \
# 	$exp/tri3_ali \
# 	$exp/nnet3/easy/egs \
# 	|| exit 1;

# fi

# if [ $stage -le 7 ]; then

#     echo "### ================ ###"
#     echo "### BEGIN TRAIN NNET ###"
#     echo "### ================ ###"

#     $steps/nnet3/train_raw_dnn.py \
#         --cmd="$train_cmd" \
#         --trainer.num-epochs 1 \
#         --trainer.optimization.num-jobs-initial=1 \
#         --trainer.optimization.num-jobs-final=1 \
#         --trainer.optimization.initial-effective-lrate=0.0015 \
#         --trainer.optimization.final-effective-lrate=0.00015 \
#         --trainer.optimization.minibatch-size=256,128 \
#         --trainer.samples-per-iter=16000 \
#         --trainer.max-param-change=2.0 \
#         --trainer.srand=0 \
#         --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
#         --feat-dir $data/train \
#         --egs.dir $exp/nnet3/easy/egs \
#         --use-dense-targets false \
#         --targets-scp $exp/tri3_ali \
#         --cleanup.remove-egs true \
#         --use-gpu false \
#         --dir=$exp/nnet3/easy  \
#         || exit 1;

    
#     # Get training ACC in right format for plotting
#     utils/format_accuracy_for_plot.sh "exp/nnet3/easy/log" "ACC_nnet3_easy.txt";

#     nnet3-am-init $exp/tri3_ali/final.mdl $exp/nnet3/easy/final.raw $exp/nnet3/easy/final.mdl || exit 1;

#     echo "### ============== ###"
#     echo "### END TRAIN NNET ###"
#     echo "### ============== ###"

# fi


# if [ $stage -le 8 ]; then

#     echo "### ============== ###"
#     echo "### BEGIN DECODING ###"
#     echo "### ============== ###"

#     test_data_dir=$data/test
#     graph_dir=$exp/tri3/graph
#     decode_dir=$exp/nnet3/easy/decode
#     final_model=$exp/nnet3/easy/final.mdl

#     mkdir -p $decode_dir

#     unknown_phone="SPOKEN_NOISE"
#     silence_phone="SIL"
    
#     $steps/nnet3/decode.sh \
#         --nj $njobs \
#         --cmd $decode_cmd \
#         --max-active 250 \
#         --min-active 100 \
#         $graph_dir \
#         $test_data_dir\
#         $decode_dir \
#         | tee $decode_dir/decode.log
#         $final_model \
#         $unknown_phone \
#         $silence_phone \

#     printf "\n#### BEGIN CALCULATE WER ####\n";
 
#     for x in ${decode_dir}*; do
#         [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh > WER_nnet3_easy.txt;
#     done

#     echo "hidden_dim=$hidden_dim"  >> WER_nnet3_easy.txt;
#     echo "num_epochs=$num_epochs"  >> WER_nnet3_easy.txt;

#     echo ""  >> WER_nnet3_easy.txt;

#     echo "test_data_dir=$test_data_dir" >> WER_nnet3_easy.txt;
#     echo "graph_dir=$graph_dir" >> WER_nnet3_easy.txt;
#     echo "decode_dir=$decode_dir" >> WER_nnet3_easy.txt;
#     echo "final_model=$final_model" >> WER_nnet3_easy.txt;
    
          
#     num_targets=`tree-info $exp/tri3_ali/tree 2>/dev/null | grep num-pdfs | awk '{print $2}'` || exit 1;

#     echo "
#     ###### BEGIN EXP INFO ######
#     num_targets= $num_targets
#     data_dir= $data_dir
#     ali_dir= $exp/tri3_ali
#     egs_dir= $exp/nnet3/easy/egs
#     ###### END EXP INFO ######
#     " >> WER_nnet3_easy.txt;

#     echo "###==============###"
#     echo "### END DECODING ###"
#     echo "###==============###"

# fi

# date --date "7 hour"
