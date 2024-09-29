MODEL=MGScoliosis 
DROP_PATH=0.2 
CUDA_VISIBLE_DEVICES=0 bash distributed_train.sh 1 'dataset_root' \
	  --model $MODEL -b 56 --lr 3e-4 --drop-path $DROP_PATH
