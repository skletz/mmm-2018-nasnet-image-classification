# Image Classification with TensorFlow-Slim NASNet

The NASNet-A model is available as TensorFlow-Slim implementation and can be found in the [research model repository ](https://github.com/tensorflow/models/tree/master/research/slim/nets/nasnet) provided by TensorFlow.

### Dependencies & Requirements
* tensorflow or tensorflow-gpu; tested with 1.11 and installed via ([tensorflow.install.pip](https://www.tensorflow.org/install/pip))
* matlibplot 2.2.3 (or higher)
* clone TensorFlow [research repository](https://github.com/tensorflow/models)

### How to run
Clone this repository as well as the TensorFlow model repository and save both in the same subdirectory. Rename the model directory of TensorFlow from "models" to "tf_models".

```bash
project
+-- tf_nasnet
+-- tf_models
|   +-- research
|   |   +-- slim
|   |   |   +-- nets
```
To execute classifications, both directories "tf_nasnet" and "tf_models" have to be added to the default search path of python. 

```bash
TF_PROJECT_DIR=/tmp/tf_nasnet
TF_MODELS_DIR=/tmp/tf_models

export PYTHONPATH="${PYTHONPATH}:${TF_PROJECT_DIR}:${TF_MODELS_DIR}/research/slim:${TF_MODELS_DIR}/research/slim/nets"

python tf_build_data.py \
--dataset_dir="./data/keyframes" \
--output_dir="./data/tfrecords" \

#Server
TF_PROJECT_DIR=/home/sabrina/Projects/tf_nasnet
TF_MODELS_DIR=/home/sabrina/Projects/models

nohup python tf_build_data.py \
--dataset_dir="/data_ssd2/uniforms" \
--output_dir="/data_ssd2/tfrecords" \
--num_threads=1 \
--shards=1 &

nohup python tf_classify_data.py \
--checkpoint_path=/home/sabrina/Projects/data/nasnet_imagenet/model.ckpt \
--input=/data_ssd2/tfrecords/shard-00000-of-00001 \
--tfrecord=True \
--output=/data_ssd2/uniforms &

#Local
WDO=/tmp

TF_INRO_DIR=${WDO}/tf_models_playground/tf_nasnet
TF_MODELS_DIR=${WDO}/tf_models_playground/tf_models

nohup python tf_classify_data.py \
--checkpoint_path=${WDO}/tf_nasnet/checkpoints/nasnet-a_large_04_10_2017/model.ckpt \
--input=${WDO}/tf_nasnet/data/tfrecords/shard-00000-of-00001 \
--tfrecord=True \
--output=${WDO}/tf_nasnet/data/classifications &
```

### Data
```bash
cd uniforms

#how many videos
ls | wc -l
7.475 #directories (=videos)

#how many keyframes
find ./ -type f | wc -l
1.948.779 #files (=keyframes)

#how much space
du -sh uniforms/
75G

#since when is the process running
ps -p ${PID} -o etime
#how many files have been processed
find ./ -type f -name "*.csv" | wc -l
```

### Performance
* Converting all keyframes into one tfrecord: 1.5h
* Classifying all keyframes and save probabilities in files: 1000 files pro Minute
