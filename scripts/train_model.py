import os
import torch as th
import numpy as np
from torch_geometric.loader import DataLoader
from InterFocusGT.data.data import HeteroPDBbindDataset
from InterFocusGT.model.model2 import InterFocusGT, HeteroGraphTransformer
from InterFocusGT.model.utils import  EarlyStopping, set_random_seed, run_a_train_epoch, run_an_eval_epoch
import argparse
import logging

p = argparse.ArgumentParser()
p.add_argument('-data_dir', '--data_dir', type=lambda s: s.split(',') ,default='data\PDBbind\PDBbind_v2020_refined,data\PDBbind\PDBbind_v2020_other_PL')
p.add_argument('-prefix', '--prefix', default="")
p.add_argument('-epoch', '--epoch', default=5000, type=int)
p.add_argument('-model_path', '--model_path', default="trained_models_by")
p.add_argument('-lr', '--lr', default=3, type=float)
p.add_argument('-b', '--batch_size', default=64, type=int)
p.add_argument('-dist', '--dist_threhold', default=7., type=float)
p.add_argument('-val_dist', '--val_dist_threhold', default=5., type=float)
p.add_argument('-lig_node_fea_num', '--lig_node_fea_num', default=41, type=int)
p.add_argument('-dropout', '--dropout', default=0.15, type=float)
p.add_argument('-valnum', '--valnum', default=1500, type=int)
p.add_argument('-num_layers', '--num_layers', default=6, type=int)
input_args = p.parse_args()

args={}
args["num_epochs"] = input_args.epoch
args["batch_size"] = input_args.batch_size
args["aux_weight"] = 0.001
args['patience'] = 70
args["num_workers"] = 0
args["model_path"] =  input_args.model_path
args['mode'] = "lower"
args['lr'] = input_args.lr
args['weight_decay'] = 5
args['device'] = 'cuda'
args['seeds'] =   126
args["data_dir"] = input_args.data_dir
args["valnum"] = input_args.valnum
args["lig_node_fea_num"] = input_args.lig_node_fea_num
args["hidden_dim0"] = 128
args["hidden_dim"] = 128
args["n_gaussians"] = 10
args["dropout_rate"] =   input_args.dropout
args["dist_threhold"] = input_args.dist_threhold
args["val_dist_threhold"] = input_args.val_dist_threhold

logging.basicConfig(
    filename='{}.log'.format(args["model_path"]),
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filemode='a'
)

logger = logging.getLogger('by')
logger.addHandler(logging.StreamHandler())

logger.info(input_args)
logger.info(args)

data = HeteroPDBbindDataset()
for d in args['data_dir']:
    tmp_data = HeteroPDBbindDataset(ids="%s/%sidsresz.npy"%(d,input_args.prefix),hetero_g="%s/%shresz.bin"%(d,input_args.prefix))
    logger.info('数据 {}，样本数量{}'.format(d , len(tmp_data)))
    data.add(tmp_data)
logger.info('总样本数量{}'.format(len(data)))

casf_pdbids = [x for x in os.listdir("data/CASF-2016/coreset") if os.path.isdir("data/CASF-2016/coreset/%s"%(x))]
logger.info('casf数据集样本数量{}'.format(len(casf_pdbids)))
data.remove(casf_pdbids)
logger.info('过滤casf数据后，样本数量{}'.format(len(data)))

train_inds, val_inds = data.train_and_test_split(valnum=args["valnum"], seed=args['seeds'])
logger.info('训练样本数量/开发样本数量：{}/{}'.format(len(train_inds),len(val_inds)))

train_data = HeteroPDBbindDataset(ids=data.pdbids[train_inds],hetero_g= [data.hetero_g[idx] for idx in train_inds])
val_data = HeteroPDBbindDataset(ids=data.pdbids[val_inds],hetero_g=[data.hetero_g[idx]  for idx in  val_inds] )
logger.info('数据划分后，训练样本数量/开发样本数量：{}/{}'.format(len(train_data),len(val_data)))

hetero_model = HeteroGraphTransformer(
                                p_in_channels = 41,
                                p_edge_features=5,
                                l_in_channels = args['lig_node_fea_num'],
                                l_edge_features=10,
                                num_hidden_channels=args["hidden_dim0"],
                                activ_fn=th.nn.SiLU(),
                                transformer_residual=True,
                                num_attention_heads=4,
                                norm_to_apply='batch',
                                dropout_rate=0.15,
                                num_layers=input_args.num_layers)

model = InterFocusGT(hetero_model,
                in_channels=args["hidden_dim0"],
                hidden_dim=args["hidden_dim"],
                n_gaussians=args["n_gaussians"],
                dropout_rate=args["dropout_rate"],
                dist_threhold=args["dist_threhold"]).to(args['device'])

optimizer = th.optim.Adam(model.parameters(), lr=10**-args['lr'], weight_decay=10**-args['weight_decay'])

train_loader = DataLoader(dataset=train_data,
                            batch_size=args["batch_size"],
                            shuffle=True,
                            num_workers=args["num_workers"])

val_loader = DataLoader(dataset=val_data,
							batch_size=args["batch_size"],
							shuffle=False,
							num_workers=args["num_workers"])

stopper = EarlyStopping(patience=args['patience'], mode=args['mode'], filename= '{}.pth'.format( args["model_path"]) )
set_random_seed(args["seeds"])


for epoch in range(args["num_epochs"]):
    # Train
    total_loss_train, mdn_loss_train, atom_loss_train, bond_loss_train = run_a_train_epoch(epoch, model, train_loader, optimizer, aux_weight=args["aux_weight"], device=args["device"] )
    if np.isinf(mdn_loss_train) or np.isnan(mdn_loss_train):
        logger.info('Inf ERROR')
        break
    total_loss_val, mdn_loss_val, atom_loss_val, bond_loss_val  = run_an_eval_epoch(model, val_loader, dist_threhold=args["val_dist_threhold"], aux_weight=args["aux_weight"], device=args["device"])
    early_stop = stopper.step(total_loss_val, model)
    logger.info('epoch {:d}/{:d} total_loss_val {:.4f}, mdn_loss_val {:.4f}, atom_loss_val {:.4f}, bond_loss_val {:.4f} ,  best validation {:.4f}'.format(epoch + 1, args['num_epochs'], total_loss_val, mdn_loss_val, atom_loss_val, bond_loss_val , stopper.best_score)) #+' validation result:', validation_result)
    if early_stop:
        break
stopper.load_checkpoint(model)
total_loss_val, mdn_loss_val, atom_loss_val, bond_loss_val= run_an_eval_epoch(model, val_loader, dist_threhold=args["val_dist_threhold"], aux_weight=args["aux_weight"], device=args["device"])
logger.info(
	"total_loss_val:%s, mdn_loss_val:%s, atom_loss_val:%s, bond_loss_val:%s" % (total_loss_val, mdn_loss_val,
																				atom_loss_val, bond_loss_val))
total_loss_train, mdn_loss_train, atom_loss_train, bond_loss_train = run_an_eval_epoch(model, train_loader, dist_threhold=args["val_dist_threhold"], aux_weight=args["aux_weight"], device=args["device"])
logger.info("total_loss_train:%s, mdn_loss_train:%s, atom_loss_train:%s, bond_loss_train:%s"%(total_loss_train, mdn_loss_train, atom_loss_train, bond_loss_train))








