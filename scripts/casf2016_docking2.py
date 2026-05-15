import numpy as np
import torch as th
from joblib import Parallel, delayed
import pandas as pd
import os
import pickle
from torch_geometric.loader import DataLoader
from InterFocusGT.data.data import HeteroVSDataset
from InterFocusGT.model.utils import run_an_eval_epoch
from InterFocusGT.model.model2 import InterFocusGT, HeteroGraphTransformer
import argparse

p = argparse.ArgumentParser()
p.add_argument('-model_name', '--model_name', default="")
p.add_argument('-model_dir', '--model_dir', default="trained_models_by")
p.add_argument('-lig_node_fea_num', '--lig_node_fea_num', default=41, type=int)
p.add_argument('-batch_size', '--batch_size', default=16, type=int)
p.add_argument('-usH', '--useH', default=False, action="store_true",
			   help='whether to use the explicit H atoms.')
p.add_argument('-uschi', '--use_chirality', default=True, action="store_true",
			   help='whether to use chirality.')
p.add_argument('-outprefix', '--outprefix', default="default")
p.add_argument("-pl_c"  , '--pl_c'   , default= 7. , type =  float)
p.add_argument("-dist_threhold"  , '--dist_threhold'   , default= 5. , type =  float)
input_args = p.parse_args()
args={}
args["batch_size"] = input_args.batch_size
args["aux_weight"] = 0.001
args["dist_threhold"] = input_args.dist_threhold
args['device'] = 'cuda'
args['seeds'] = 126
args["num_workers"] = 0
args["model_path"] = "{}/{}".format(input_args.model_dir,input_args.model_name)
args["cutoff"] = 10.0
args["num_node_featsp"] = 41
args["num_node_featsl"] = input_args.lig_node_fea_num
args["num_edge_featsp"] = 5
args["num_edge_featsl"] = 10
args["hidden_dim0"] = 128 
args["hidden_dim"] = 128
args["n_gaussians"] = 10
args["dropout_rate"] = 0.10
args["outprefix"] = input_args.outprefix

outdir = "output_dir\%s"%args["outprefix"]
cmd = "mkdir  %s"%outdir
os.system(cmd)

def scoring(prot, lig, modpath,
			cut=10.0,
			pl_c = 7. ,
			explicit_H=False, 
			use_chirality=True,
			parallel=False,
			**kwargs
			):
	data = HeteroVSDataset(ligs=lig,
					prot=prot,
					cutoff=cut,
					pl_c = pl_c ,
					explicit_H=explicit_H, 
					use_chirality=use_chirality,
					parallel=parallel)
	test_loader = DataLoader(dataset=data, 
							batch_size=kwargs["batch_size"],
							shuffle=False, 
							num_workers=kwargs["num_workers"])
	hetero_model = HeteroGraphTransformer(
		p_in_channels=kwargs["num_node_featsp"],
		p_edge_features=kwargs["num_edge_featsp"],
		l_in_channels=kwargs["num_node_featsl"],
		l_edge_features=kwargs["num_edge_featsl"],
		num_hidden_channels=kwargs["hidden_dim0"],
		activ_fn=th.nn.SiLU(),
		transformer_residual=True,
		num_attention_heads=4,
		norm_to_apply='batch',
		dropout_rate=0.15,
		num_layers=6)

	model = InterFocusGT(hetero_model,
					in_channels=kwargs["hidden_dim0"], 
					hidden_dim=kwargs["hidden_dim"], 
					n_gaussians=kwargs["n_gaussians"], 
					dropout_rate=kwargs["dropout_rate"], 
					dist_threhold=kwargs["dist_threhold"]).to(kwargs['device'])

	checkpoint = th.load(modpath, map_location=th.device(kwargs['device']))
	model.load_state_dict(checkpoint['model_state_dict'])
	preds = run_an_eval_epoch(model, test_loader, pred=True, dist_threhold=kwargs['dist_threhold'], device=kwargs['device'])
	return data.ids, preds


def score_compound(pdbid, prefix):
	return scoring(
					prot="data/PDBbind/PDBbind_%s/%s/%s_pocket_10_preprocessed_1124.pdb" %(prefix ,pdbid, pdbid ) ,
					lig="data/CASF-2016/decoys_docking/%s_decoys.mol2"%pdbid,
					modpath= args["model_path"],
					cut=args["cutoff"],
				    pl_c= input_args.pl_c ,
					explicit_H=input_args.useH,
					use_chirality=input_args.use_chirality,
					parallel=False,
					**args
					)

def score_compound0(pdbid, prefix):
	ids, scores = scoring(
					prot="data/PDBbind/PDBbind_%s/%s/%s_pocket_10_preprocessed_1124.pdb" %(prefix ,pdbid, pdbid ),
					lig="data/PDBbind/PDBbind_%s/%s/%s_ligand.mol2"%(prefix, pdbid, pdbid),
					modpath= args["model_path"],
					cut=args["cutoff"],
				    pl_c = input_args.pl_c,
				    explicit_H=input_args.useH,
					use_chirality=input_args.use_chirality,
					parallel=False,
					**args
					)
	ids.pop(-1)
	ids.append("%s_ligand"%pdbid)
	return ids, scores

def score_compoundxxx(pdbid, prefix):
	ids1, scores1 = score_compound(pdbid, prefix)
	ids2, scores2 = score_compound0(pdbid, prefix)
	return ids1+ids2, np.append(scores1,scores2)

pdbids = [x for x in os.listdir("data/CASF-2016/coreset") if os.path.isdir("data/CASF-2016/coreset/%s"%(x))]
pdbids1 = [x for x in os.listdir("data/PDBbind/PDBbind_v2020_refined") if os.path.isdir("data/PDBbind/PDBbind_v2020_refined/%s"%(x))]
pdbids2 = [x for x in os.listdir("data/PDBbind/PDBbind_v2020_other_PL") if os.path.isdir("data/PDBbind/PDBbind_v2020_other_PL/%s"%(x))]

print('pdbids',len(pdbids))
print('pdbids1' ,len(pdbids1))
print('pdbids2' ,len(pdbids2))

ids1 = [pdbid for pdbid in pdbids if pdbid in pdbids1]
ids2 = [pdbid for pdbid in pdbids if pdbid in pdbids2]
print('ids1' ,len(ids1))
print('ids2' ,len(ids2))

if args['device'] == 'cpu':
	results1 = Parallel(n_jobs=-1)(delayed(score_compoundxxx)(pdbid, "v2020_refined") for pdbid in ids1)
	results2 = Parallel(n_jobs=-1)(delayed(score_compoundxxx)(pdbid, "v2020_other_PL") for pdbid in ids2)
	results = results1 + results2
else:
	print("使用GPU评测")
	results = []
	idx = 0
	for pdbid in ids1:
		idx+=1
		print('scoring {}/{}'.format(idx, pdbid))
		results.append(score_compoundxxx(pdbid, "v2020_refined"))

	for pdbid in ids2:
		idx+=1
		print('scoring {}/{}'.format(idx, pdbid))
		results.append(score_compoundxxx(pdbid, "v2020_other_PL"))

for res in results:
	pdbid = res[0][0].split("_")[0]
	df = pd.DataFrame(zip(*res),columns=["#code","score"])
	df["#code"] = df["#code"].str.split("-").apply(lambda x : x[0])
	df.to_csv("%s/%s_score.dat"%(outdir, pdbid), index=False, sep="\t")

with open(outdir+"\%s_docking.pkl"%args["outprefix"],"wb") as dbFile:
	pickle.dump(results,dbFile)

