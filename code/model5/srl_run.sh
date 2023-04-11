# sudo conda activate biosyntax
export BIOBERT_DIR=./biobert_v1.1_pubmed
echo $BIOBERT_DIR
export SRL_DIR=./srl_data
export OUTPUT_DIR=./srl_output_nopredicate
#rm -rf $OUTPUT_DIR
#mkdir -p $OUTPUT_DIR
nohup /opt/conda/bin/python run_srl.py --do_train=False --do_eval=True --do_predict=True --vocab_file=$BIOBERT_DIR/vocab.txt --bert_config_file=$BIOBERT_DIR/bert_config.json --init_checkpoint=$BIOBERT_DIR/model.ckpt-1000000 --num_train_epochs=10.0 --data_dir=$SRL_DIR --output_dir=$OUTPUT_DIR > nohup_srl_nopredicate.out &
# python biocodes/ner_detokenize.py --token_test_path=$OUTPUT_DIR/token_test.txt --label_test_path=$OUTPUT_DIR/label_test.txt --answer_path=$NER_DIR/test.tsv --output_dir=$OUTPUT_DIR
#nohup /opt/conda/bin/python run_srl.py --do_train=false --do_eval=false --do_predict=true --vocab_file=$BIOBERT_DIR/vocab.txt --bert_config_file=$BIOBERT_DIR/bert_config.json --init_checkpoint=$BIOBERT_DIR/model.ckpt-1000000 --num_train_epochs=3.0 --data_dir=$NER_DIR --output_dir=$OUTPUT_DIR &
# perl biocodes/conlleval.pl < $OUTPUT_DIR/NER_result_conll.txt 


#nohup /opt/conda/envs/biosyntax/bin/python3 run_srl.py --do_train=true --do_eval=true --do_predict=true --vocab_file=$BIOBERT_DIR/vocab.txt --bert_config_file=$BIOBERT_DIR/bert_config.json --init_checkpoint=$BIOBERT_DIR/model.ckpt-1000000 --num_train_epochs=10.0 --data_dir=$NER_DIR --output_dir=$OUTPUT_DIR &

# python3 run_BioByGANS_cpu.py --do_train=true --do_eval=true --do_predict=true --vocab_file=$BIOBERT_DIR/vocab.txt --bert_config_file=$BIOBERT_DIR/bert_config.json --init_checkpoint=$BIOBERT_DIR/model.ckpt-1000000 --num_train_epochs=10.0 --data_dir=$NER_DIR --output_dir=$OUTPUT_DIR

#python ./biocodes/ner_detokenize.py --token_test_path=$OUTPUT_DIR/token_test.txt --label_test_path=$OUTPUT_DIR/label_test.txt --answer_path=$NER_DIR/test.tsv --output_dir=$OUTPUT_DIR
#perl ./biocodes/conlleval.pl < $OUTPUT_DIR/NER_result_conll.txt
