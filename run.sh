watch -n 0.1 -d nvidia-smi
cd "/mnt/wangbolin/code/DeepSEED/"; source ./bin/activate; cd deepseed4mouse

### we take design of 3-lacO IPTG-inducible promoters in *E. coli* as an example, to illustrate how to train the DeepSEED model and design the promoter sequences

### 1. Training the conditional GANs (Expected run time on GeForce GTX 1080Ti: 10 hours)
cd Generator; 
python cGAN_training.py --data_name random_seq_v2 --batch_size 32 --seqL 1000 --gpuid 1 \
    --n_critics  5  --n_iters 10000
cd Generator;
python cGAN_training.py --data_name random_promoters_v2 --batch_size 32 --seqL 1000 --gpuid 2 \
    --n_critics  25  --n_iters 10000      

cd Generator;
python Transformer_training.py --data_name based_cst69_promoters --batch_size 256   --gpuid 0 \
    --nhead  8  --num_layers 1 --d_model 256  --n_critics 50 --n_iters 1000 --weight 0.0

### 2. Training the predictor (Expected run time on GeForce GTX 1080Ti: 0.5 hour)
cd Predictor; python predictor_training.py

### 3. Design the promoter sequences with optimizer (Expected run time on GeForce GTX 1080Ti: 24 hours)
cd Optimizer; python deepseed_optimizer.py

### 4. Check the synthetic promoter sequences!
vi ./results/ecoli_3_laco.txt





