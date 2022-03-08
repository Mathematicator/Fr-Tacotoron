    #!/bin/bash

source /applis/environments/cuda_env.sh dahu 10.1
#source /applis/environments/conda.sh
#conda activate GPU
#source NVIDIA_official/bin/activate

conda activate condaPy3.6
cd /bettik/PROJECTS/pr-esyn/tacotron2_gpu2/
#python3 -m multiproc train.py --output_directory=outdir --log_directory=logdir –c 'outdir/checkpoint_1000' --warm_start 

python3 -m multiproc train.py --output_directory=out --log_directory=log --hparams=distributed_run=True,fp16_run=True
#python3 train.py --log_directory logdir --output_directory outdir -c outdir/checkpoint_1000 --warm_start

exit 0
