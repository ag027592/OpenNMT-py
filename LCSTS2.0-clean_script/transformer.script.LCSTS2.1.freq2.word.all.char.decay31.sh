TITLE="20180504_freq2_word_all_char_transformer_decay31"
DATASET="LCSTS2.0-clean"
ENC_TYPE="word"
DEC_TYPE="char" 
PROGRAM="OpenNMT-py-latest"
METHOD=$ENC_TYPE"_"$DEC_TYPE
ORIGIN_DIR="/2t/playma/"$DATASET"/"$METHOD"_data"
DATA_DIR="/home/playma/data/"$PROGRAM"/"$DATASET"/"$METHOD"_data-bin_freq2_word_all_char"
MODEL_DIR="/2t/playma/model/"$PROGRAM"/"$DATASET"/"$METHOD"_checkpoints/"$TITLE
MODEL_NAME="_acc_61.86_ppl_9.49_e10.pt"
TEST=$ORIGIN_DIR"/test.source"
PRED_DIR=$MODEL_DIR"/pred_result"
PRED=$PRED_DIR"/"$MODEL_NAME
GOLD=$ORIGIN_DIR"/test.target"

if [ $1 = "preprocess" ]; then
    echo "Start Preprocessing."
    mkdir -p $DATA_DIR
    time python3 ../preprocess.py \
      -train_src $ORIGIN_DIR/train.source \
      -train_tgt $ORIGIN_DIR/train.target \
      -valid_src $ORIGIN_DIR/valid.source \
      -valid_tgt $ORIGIN_DIR/valid.target \
      -src_vocab_size 10000000 \
      -tgt_vocab_size 10000000 \
      -src_seq_length 1000 \
      -tgt_seq_length 1000 \
      -src_words_min_frequency 2 \
      -max_shard_size 10485760 \
      -save_data $DATA_DIR/processed
    echo "Finish Preprocessing"
elif [ $1 = "train" ]; then
    echo "Start training."
    mkdir -p $MODEL_DIR

    python3 ../train.py \
        -data $DATA_DIR/processed \
        -save_model $MODEL_DIR/ \
        -train_from $MODEL_DIR/$MODEL_NAME \
        -gpuid 1 \
        -layers 4 \
        -rnn_size 512 \
        -word_vec_size 512 \
        -encoder_type transformer \
        -decoder_type transformer \
        -position_encoding \
        -epochs 50 \
        -max_generator_batches 32 \
        -dropout 0.1 \
        -batch_size 4096 \
        -batch_type tokens \
        -normalization tokens \
        -accum_count 4 \
        -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
        -max_grad_norm 0 \
        -param_init 0 \
        -param_init_glorot \
        -label_smoothing 0.1 \
        -start_decay_at 31 \
        > $MODEL_DIR/log.con.2.txt

elif [ "$1" = "generate" ]; then
    echo "Start gnerate interactive."
    python3 translate.py \
      -model $MODEL_DIR/$MODEL_NAME \
      -beam_size 5 \
      -verbose \
      -batch_size 1 \
      -tgt $GOLD \
      -output $PRED \
      -src $TEST \
      -report_rouge
elif [ "$1" = "generate_all" ]; then
    mkdir -p $PRED_DIR
    i=1
    for file in `ls -tr $MODEL_DIR/_*`
    do
        echo $file
        filebasename=$(basename "${file%.*}")
        output_path=$PRED_DIR/$filebasename.pred.txt
        if [ -e $output_path ]
        then
            continue
        fi
        python3 ../translate.py \
          -model $file \
          -beam_size 5 \
          -verbose \
          -gpu 0 \
          -output $output_path \
          -src $TEST
        i=$((i+1))
    done
elif [ "$1" = "evaluate" ]; then
    echo "Start evaluate."
    echo "PRED path: "$PRED
    echo "GOLD path: "$GOLD
    perl ROUGE_with_ranked.pl 1 N $GOLD $PRED
    perl ROUGE_with_ranked.pl 2 N $GOLD $PRED R
    perl ROUGE_with_ranked.pl 1 L $GOLD $PRED
elif [ "$1" = "evaluate_all" ]; then
    i=1
    for file in `ls -tr $PRED_DIR/_*`
    do
        filebasename=$(basename "${file%.*}")
        echo $filebasename
        perl ../ROUGE_with_ranked.pl 1 N $GOLD $file
        perl ../ROUGE_with_ranked.pl 2 N $GOLD $file R
        perl ../ROUGE_with_ranked.pl 1 L $GOLD $file
        echo "\n=========================================="
        i=$((i+1))
    done
fi

echo "Done"
