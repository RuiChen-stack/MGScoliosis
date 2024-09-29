MODEL=MGScoliosis 
python3 validate.py 'dataset_root' --model $MODEL \
  --checkpoint 'checkpoint.pth' -b 64
