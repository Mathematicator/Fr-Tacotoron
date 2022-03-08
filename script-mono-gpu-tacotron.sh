    #!/bin/bash

source /applis/environments/cuda_env.sh dahu 10.1
source /applis/environments/conda.sh
#conda activate GPU
source NVIDIA_official/bin/activate

cd /bettik/PROJECTS/pr-esyn/tacotron2_gpu2/
python3 train.py --log_directory logdir --output_directory outdir -c outdir/checkpoint_1000 --warm_start

exit 0
# python train.py --log_directory logdir_test --output_directory outdir_test -c outdir/checkpoint_1000 --warm_start