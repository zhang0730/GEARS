import os #用于处理文件路径
import time #计时
import argparse #用于解析命令行参数。
import pandas as pd #用于数据处理和保存参数。
import scanpy as sc #用于处理单细胞数据（.h5ad 格式）。
from os.path import join as pjoin #

from gears import PertData, GEARS #主体还是gears原有的包 分别用于数据处理和模型训练。

#直接上来就是主函数
def main(parser):
    args = parser.parse_args() #解析命令行参数，并将参数存储在 args 对象中。

    # get data 加载数据
    pert_data = PertData(args.data_dir) #创建 PertData 对象，用于加载和处理数据。
    # load dataset in paper: norman, adamson, dixit.
    try:
        if args.data_name in ['norman', 'adamson', 'dixit']: #如果数据名是 norman、adamson 或 dixit，则加载预定义的数据集。
            pert_data.load(data_name = args.data_name)
        else:
            print('load data')
            pert_data.load(data_path = pjoin(args.data_dir, args.data_name))
    except: #否则，尝试从指定路径加载数据。如果加载失败，则使用 scanpy 读取 .h5ad 文件，并进行数据预处理。
        adata = sc.read_h5ad(pjoin(args.data_dir, args.data_name+'.h5ad'))
        adata.uns['log1p'] = {}
        adata.uns['log1p']['base'] = None
        pert_data.new_data_process(dataset_name=args.data_name, adata=adata)
        
    #数据划分器 specify data split  根据指定的划分方式（split）和随机种子（seed）划分数据集。train_gene_set_size 控制训练集中基因集的大小。
    pert_data.prepare_split(split = args.split, seed = args.seed, train_gene_set_size=args.train_gene_set_size)
    # get dataloader with batch size  获取训练和测试数据的数据加载器，设置训练和测试的批次大小。
    pert_data.get_dataloader(batch_size = args.batch_size, test_batch_size = args.test_batch_size)

    # set up and train a model 初始化模型
    gears_model = GEARS(pert_data, device = args.device)
    gears_model.model_initialize(hidden_size = args.hidden_size,  #设置隐藏层大小
                                 model_type = args.model_type, #模型类型
                                 bin_set=args.bin_set, #是否使用二进制特征
                                 load_path=args.singlecell_model_path, #预训练模型路径 这个是重点就是承接scFoundation和gears的地方
                                 finetune_method=args.finetune_method, #微调方法
                                 accumulation_steps=args.accumulation_steps, #梯度累积步数
                                 mode=args.mode, #模型模式
                                 highres=args.highres) #是否使用高分辨率数据

    #训练模型，设置训练轮数（epochs）、结果保存目录（result_dir）和学习率（lr）。
    gears_model.train(epochs = args.epochs, result_dir=args.result_dir,lr=args.lr)

    # save model 将训练好的模型保存到指定目录。
    gears_model.save_model(args.result_dir)

    # save params 将命令行参数保存为 params.csv 文件，便于后续查看和复现实验。
    param_pd = pd.DataFrame(vars(args), index=['params']).T
    param_pd.to_csv(f'{args.result_dir}/params.csv')

if __name__ == '__main__': #用于确保某些代码块只在脚本直接运行（例如 python train.py）时执行，而在脚本被作为模块导入（例如 import train）时不执行。
    parser = argparse.ArgumentParser(description='GEARS')

    parser.add_argument('--data_dir', type=str, default='./data') #数据目录，默认为 ./data。
    parser.add_argument('--data_name', type=str, default='norman') #数据集名称，默认为 norman。
    parser.add_argument('--split', type=str, default='simulation') #数据划分方式，默认为 simulation。
    parser.add_argument('--result_dir', type=str, default='./results') #结果保存目录，默认为 ./results。
    parser.add_argument('--seed', type=int, default=1) #随机种子，默认为1。
    parser.add_argument('--epochs', type=int, default=20) #训练轮数，默认为20。
    parser.add_argument('--batch_size', type=int, default=32) #训练批次大小，默认为32。
    parser.add_argument('--test_batch_size', type=int, default=128) #测试批次大小，默认为128。
    parser.add_argument('--train_gene_set_size', type=float, default=0.75) #训练集基因集大小，默认为0.75。
    parser.add_argument('--hidden_size', type=int, default=64) #模型隐藏层大小，默认为64。
    parser.add_argument('--device', type=str, default='cuda') #训练设备，默认为 cuda（GPU）。
    parser.add_argument('--model_type', type=str, default=None) #模型类型，默认为 None。
    parser.add_argument('--bin_set', type=str, default=None) #是否使用二进制特征，默认为 None。
    parser.add_argument('--singlecell_model_path', type=str, default=None) #预训练模型路径，默认为 None。这个的自己输入
    parser.add_argument('--finetune_method', type=str, default=None) #微调方法，默认为 None。
    parser.add_argument('--mode', type=str, default='v1') #模型模式，默认为 v1。
    parser.add_argument('--accumulation_steps', type=int, default=1) #梯度累积步数，默认为1。
    parser.add_argument('--highres', type=int, default=0) #是否使用高分辨率数据，默认为0。
    parser.add_argument('--lr', type=float, default=1e-3) #学习率，默认为 1e-3。
    

    main(parser)