    #!/bin/bash

source /applis/environments/cuda_env.sh dahu 10.1
#source /applis/environments/conda.sh
#conda activate GPU
#source NVIDIA_official/bin/activate

conda activate condaPy3.6

cd /bettik/PROJECTS/pr-esyn/tacotron2_gpu2/
python3 train.py --log_directory logdir --output_directory outdir -c outdir/checkpoint_1000 --warm_start --hparams=batch_size=32

exit 0
