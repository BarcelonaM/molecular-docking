import torch as th
from joblib import Parallel, delayed
import pandas as pd
import os
from torch_geometric.loader import DataLoader
from InterFocusGT.data.data import HeteroVSDataset ,build_seed_hg
from InterFocusGT.model.utils import run_an_eval_epoch
from InterFocusGT.model.model2 import InterFocusGT, HeteroGraphTransformer
import torch.multiprocessing
from joblib import load, dump
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
p = argparse.ArgumentParser()
p.add_argument('-model_name', '--model_name', default="")
p.add_argument('-model_dir', '--model_dir', default="trained_models_by")
p.add_argument('-lig_node_fea_num', '--lig_node_fea_num', default=41, type=int)
p.add_argument('-batch_size', '--batch_size', default=128, type=int)
p.add_argument('-usH', '--useH', default=False, action="store_true",
			   help='whether to use the explicit H atoms.')
p.add_argument('-uschi', '--use_chirality', default=False, action="store_true",
			   help='whether to use chirality.')
p.add_argument('-outprefix', '--outprefix', default="test")
p.add_argument("-pl_c"  , '--pl_c'   , default= 7. , type =  float)
p.add_argument('-only_features', '--only_features', default=False, action="store_true",)
p.add_argument('-device', '--device', default="cpu")
p.add_argument('-parallel', '--parallel', default=False, action="store_true",)
input_args = p.parse_args()
args={}
args["batch_size"] = input_args.batch_size
args["aux_weight"] = 0.001
args["dist_threhold"] = 5
args['device'] =  input_args.device
args['seeds'] = 126
args["num_workers"] = 0
args["model_path"] = "{}/{}".format(input_args.model_dir,input_args.model_name)
args["cutoff"] = 10
args["num_node_featsp"] = 41
args["num_node_featsl"] = input_args.lig_node_fea_num
args["num_edge_featsp"] = 5
args["num_edge_featsl"] = 10
args["hidden_dim0"] = 128
args["hidden_dim"] = 128
args["n_gaussians"] = 10
args["dropout_rate"] = 0.15
args["outprefix"] = input_args.outprefix
outdir = "output_dir\%s_screening"%args["outprefix"]

has_scored = set()
if  os.path.isdir(outdir):
	print("{}文件夹存在".format(outdir))
	for pid in os.listdir(outdir):
		has_scored.add(pid.split('_')[0])
else:
	cmd = "mkdir  %s"%outdir
	os.system(cmd)

print('has scored pdb id ',has_scored)

def scoring(prot, lig, modpath,
			cut=10.0,
			pl_c=7. ,
			explicit_H=False, 
			use_chirality=True,
			parallel=False,
			seed_hg = None ,
			**kwargs
			):

	if not input_args.only_features:
		pre, ext = os.path.splitext(lig)
		data_path = pre + '.bin'
		data = load(data_path)
		print('loaded' , data_path)

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

	else:
		pre, ext = os.path.splitext(lig)
		data_path = pre+'.bin'

		if os.path.exists(data_path):
			os.remove(data_path)

		data = HeteroVSDataset(ligs=lig,
							   prot=prot,
							   cutoff=cut,
							   pl_c=pl_c,
							   explicit_H=explicit_H,
							   use_chirality=use_chirality,
							   parallel=parallel,
							   seed_hg=seed_hg)

		dump(data, data_path)
		print('dumped' , data_path)
		return None , None

def score_compound(pdbid, ligid, prefix , seed_hg , **args):
	return scoring(prot="data/PDBbind/PDBbind_%s/%s/%s_pocket_10_preprocessed_1124.pdb" % (prefix, pdbid, pdbid),
					lig="data/CASF-2016/decoys_screening/%s/%s_%s.mol2"%(pdbid, pdbid, ligid),
					modpath=args["model_path"],
					cut=args["cutoff"],
				   	pl_c=input_args.pl_c,
				    explicit_H=input_args.useH,
				    use_chirality=input_args.use_chirality,
					parallel=input_args.parallel,
				   seed_hg = seed_hg,
					**args
					)


def score_compound2(idx,  pdbid, prefix, ligids, **args):
	print('当前处理第{}个蛋白质'.format(idx ))
	ids_list = []
	scores_list = []
	seed_hg = build_seed_hg(
		prot="data/PDBbind/PDBbind_%s/%s/%s_pocket_10_preprocessed_1124.pdb" % (prefix, pdbid, pdbid),
											 cutoff=args["cutoff"],
											 reflig=None,
											 gen_pocket=False,
											 explicit_H=input_args.useH,
											 use_chirality=input_args.use_chirality
											 )
	for i , ligid in  enumerate (ligids):
		ids, scores = score_compound(pdbid, ligid, prefix,seed_hg, **args)
		if ids is not None and scores is not None:
			ids_list.extend(ids)
			scores_list.extend(scores)
	return pdbid, [ids_list, scores_list]

def save_pdb_score(res , tmpoutdir):
	if not input_args.only_features:
		pdbid = res[0]
		df = pd.DataFrame(zip(*res[1]),columns=["#code_ligand_num","score"])
		df["#code_ligand_num"] = df["#code_ligand_num"].str.split("-").apply(lambda x : x[0])
		df.to_csv("%s/%s_score.dat"%(tmpoutdir, pdbid), index=False, sep="\t")

ligids = [x for x in os.listdir('data/CASF-2016/coreset') if os.path.isdir("data/CASF-2016/coreset/%s"%(x))]
pdbids = [x for x in os.listdir("data/CASF-2016/decoys_screening") if os.path.isdir("data/CASF-2016/decoys_screening/%s"%(x))]
pdbids1 = [x for x in os.listdir("data/PDBbind/PDBbind_v2020_refined") if os.path.isdir("data/PDBbind/PDBbind_v2020_refined/%s"%(x))]
pdbids2 = [x for x in os.listdir("data/PDBbind/PDBbind_v2020_other_PL") if os.path.isdir("data/PDBbind/PDBbind_v2020_other_PL/%s"%(x))]

print('pdbids' , len(pdbids))
print('pdbids1' , len(pdbids1))
print('pdbids2' , len(pdbids2))

ids1 = [pdbid for pdbid in pdbids if pdbid in pdbids1]
ids2 = [pdbid for pdbid in pdbids if pdbid in pdbids2]

print('total:{} refined:{} other:{}'.format(len(pdbids) ,len(ids1) , len(ids2)))
print('ligids' , len(ligids))

if args['device'] == 'cpu':
	results1 = Parallel(n_jobs= 1)(delayed(score_compound2)(i ,pdbid, "v2020_refined", ligids, **args) for i , pdbid in  enumerate(ids1))
	results2 = Parallel(n_jobs= 1)(delayed(score_compound2)(i , pdbid, "v2020_other_PL", ligids, **args) for  i,  pdbid in enumerate( ids2))
	results = results1 + results2
	for r in results :
		save_pdb_score(r,outdir)
else:
	print('use gpu device')
	results = []
	idx = 0
	for pdbid in ids1:
		idx+=1
		results.append(score_compound2(idx , pdbid, "v2020_refined", ligids, **args))
		save_pdb_score(results[-1],outdir)
	for pdbid in ids2:
		idx += 1
		results.append(score_compound2(idx , pdbid, "v2020_other_PL", ligids, **args))
		save_pdb_score(results[-1],outdir)
