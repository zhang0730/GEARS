#用于训练 GEARS 模型的 Bash 脚本。它配置了训练过程中的各种参数，并调用 Python 脚本 train.py 来执行训练任务。

#(1) 设置脚本执行模式 set -x：在执行时打印每一行命令，方便调试。
set -xe

#指定使用的 GPU 设备。默认使用 GPU 0。
device_id=0 # which device to run the program, for multi-gpus, set params like device_id=0,2,5,7. [Note that] the device index in python refers to 0,1,2,3 respectively.

# params 设置训练参数
data_dir=./data/ #数据目录路径。
data_name=demo #数据集名称，这里使用的是 demo。
split=simulation #数据分割方式，这里使用的是 simulation。
result_dir=./results #结果保存目录。
seed=1 #随机种子，用于确保实验的可重复性。
epochs=1 #训练的轮数。
batch_size=2 #训练时的批次大小。
accumulation_steps=1 #梯度累积的步数。
test_batch_size=2 #测试时的批次大小。
hidden_size=512 #模型的隐藏层大小。
train_gene_set_size=0.75 #训练时使用的基因集大小比例。
mode=v1 #训练模式，这里使用的是 v1。
highres=0 #是否使用高分辨率模式，0 表示不使用。
lr=0.0002 #1e-3 学习率。

#设置模型参数
model_type=maeautobin #模型类型，这里使用的是 maeautobin。
bin_set=autobin_resolution_append #autobin_resolution, bin_2, bin_3, no_bin 基因分箱设置，这里使用的是 autobin_resolution_append。
finetune_method='frozen' # [None,finetune, 'frozen', 'finetune_lr_1']) 微调方法，这里使用的是 frozen（冻结模型参数）。
singlecell_model_path=../model/models/models.ckpt #预训练的 scFoundation 模型路径。
workdir=./
cd $workdir #将工作目录切换到当前目录。

export TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S") #生成当前时间戳，用于唯一标识本次实验。

result_dir=${workdir}/results/${data_name}/${train_gene_set_size}/50m-0.1B_split_${split}_seed_${seed}_hidden_${hidden_size}_bin_${bin_set}_singlecell_${model_type}_finetune_${finetune_method}_epochs_${epochs}_batch_${batch_size}_accmu_${accumulation_steps}_mode_${mode}_highres_${highres}_lr_${lr}/${TIMESTAMP}/

mkdir -p ${result_dir} #根据参数生成结果目录路径。

#运行训练脚本 指定使用的 GPU 设备 运行 train.py 脚本，-u 参数确保输出无缓冲。
CUDA_VISIBLE_DEVICES=${device_id} python -u train.py \
    --data_dir=${data_dir} \
    --data_name=${data_name} \
    --seed=${seed} \
    --result_dir=${result_dir} \
    --seed=${seed} \
    --epochs=${epochs} \
    --batch_size=${batch_size} \
    --test_batch_size=${test_batch_size} \
    --hidden_size=${hidden_size} \
    --bin_set=${bin_set} \
    --model_type=${model_type} \
    --finetune_method=${finetune_method} \
    --singlecell_model_path=${singlecell_model_path} \
    --mode=${mode} \
    --highres=${highres} \
    --accumulation_steps=${accumulation_steps} \
    --lr=${lr} > ${result_dir}/train.log 2>&1
#

# --singlecell_model_path=${singlecell_model_path}