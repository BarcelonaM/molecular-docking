from torch.utils.data import Dataset
import numpy as np
from rdkit import Chem
from joblib import Parallel, delayed
import os
from ..feats.mol2graph_rdmda_res import mol_to_graph, load_mol, prot_to_graph ,ligand_process_without_kekulize , create_hetero_graph
from torch_geometric.data import HeteroData
from joblib import load
import traceback
class  HeteroPDBbindDataset(Dataset):
	def __init__(self,
				 ids=None,
				 hetero_g=None
				 ):
		if ids is None and hetero_g is None :
			self.pdbids = None
			self.hetero_g = []
			return
		if isinstance(ids, np.ndarray) or isinstance(ids, list):
			self.pdbids = ids
		else:
			try:
				self.pdbids = np.load(ids)
			except:
				raise ValueError('the variable "ids" should be numpy.ndarray or list or a file to store numpy.ndarray')
		if isinstance(hetero_g, np.ndarray) or isinstance(hetero_g, tuple) or isinstance(hetero_g, list):
			if isinstance(hetero_g[0],  HeteroData ):
				self.hetero_g = hetero_g
			else:
				raise ValueError('the variable "ligs" should be a set of (or a file to store) dgl.DGLGraph objects.')
		else:
			try:
				self.hetero_g  = load(hetero_g )
			except:
				raise ValueError('the variable "ligs" should be a set of (or a file to store) dgl.DGLGraph objects.')
		self.hetero_g = list(self.hetero_g)
		assert len(self.pdbids) == len(self.hetero_g)

	def __getitem__(self, idx):
		return self.pdbids[idx], self.hetero_g[idx]

	def __len__(self):
		return len(self.pdbids) if self.pdbids is not None else 0

	def add(self , data):
		if  self.pdbids is None :
			self.pdbids = data.pdbids
		else:
			self.pdbids = np.concatenate([self.pdbids,data.pdbids] ,axis=0)
		self.hetero_g.extend(data.hetero_g)
		assert len(self.pdbids) == len(self.hetero_g)

	def remove(self,pids):
		pids = set(pids)
		mask = []
		for idx in range(len(self.pdbids)):
			if self.pdbids[idx] in pids:
				mask.append(False)
			else:
				mask.append(True)
		self.pdbids = self.pdbids[mask]
		self.hetero_g = [e for e, m in zip(self.hetero_g, mask) if m]
		assert len(self.pdbids) == len(self.hetero_g)

	def train_and_test_split(self, valfrac=0.2, valnum=None, seed=0):
		np.random.seed(seed)
		if valnum is None:
			valnum = int(valfrac * len(self.pdbids))
		val_inds = np.random.choice(np.arange(len(self.pdbids)),valnum, replace=False)
		train_inds = np.setdiff1d(np.arange(len(self.pdbids)),val_inds)
		return train_inds, val_inds

	def train_and_test_split_refined(self,valfrac = 0.2 ,  valnum= None , seed= 0 ):
		refined_pdb_ids = set(os.listdir('data/PDBbind/PDBbind_v2020_refined'))
		refined_choice = [  idx   for idx , pdb_id in enumerate (self.pdbids) if pdb_id in refined_pdb_ids]
		if valnum is None:
			valnum = int(valfrac * len(refined_choice))
		np.random.seed(seed)
		val_inds = np.random.choice( refined_choice ,valnum, replace=False)
		train_inds = np.setdiff1d(np.arange(len(self.pdbids)),val_inds)
		return train_inds, val_inds

class HeteroVSDataset(Dataset):
	def __init__(self,  
				ids=None,
				ligs=None,
				prot=None,
				gen_pocket=False,
				cutoff=None,
				pl_c = 7.,
				reflig=None,
				explicit_H=False, 
				use_chirality=True,
				parallel=True,
				seed_hg = None,
				flag =  '',
				 range_begin = -1,
				 steps = -1,

				):

		self.cutoff = cutoff
		self.explicit_H=explicit_H
		self.use_chirality=use_chirality
		self.parallel=parallel
		self.hetero_graphs = []
		self.ids = []

		if ligs is None and prot is None:
			return

		if seed_hg is None : # 制造种子

			if isinstance(prot, list) :
				assert isinstance(prot[0], Chem.rdchem.Mol)
				def tmp_build_seed_hg(tmp_prot):
					try:
						tmp_seed_hg = HeteroData()
						prot_to_graph(tmp_prot, cutoff, tmp_seed_hg)
						return tmp_seed_hg
					except Exception as e:
						print('tmp_build_seed_hg error')
						traceback.print_exc()
						return None
				seed_hg  = Parallel(n_jobs= -1)(delayed(tmp_build_seed_hg) ( p )  for p in prot)
			else:
				prot_mol = load_prot(prot,explicit_H ,use_chirality)
				seed_hg = HeteroData()
				prot_to_graph(prot_mol, cutoff, seed_hg)

		ligs_mols,idsx = load_ligs(ligs,explicit_H , use_chirality ,flag  )

		if ids is not None :
			idsx = ids


		if isinstance(seed_hg , list):
			multi_graph = Parallel(n_jobs= -1)(delayed(lig_mols_to_graph_multi_process) (  lig ,explicit_H , use_chirality ,pl_c, s_hg )  for s_hg  ,  lig in zip ( seed_hg , ligs_mols ) )
		else:
			multi_graph = Parallel(n_jobs= -1)(delayed(lig_mols_to_graph_multi_process) (  lig ,explicit_H , use_chirality ,pl_c, seed_hg )  for   lig in   ligs_mols  )

		self.hetero_graphs.extend(multi_graph)
		self.ids, self.hetero_graphs = zip(*filter(lambda x: x[1] is not None, zip(idsx, self.hetero_graphs)))
		self.ids = list(self.ids)
		self.hetero_graphs = list(self.hetero_graphs)
		assert len(self.ids) == len(self.hetero_graphs)


	def __getitem__(self, idx):
		return self.ids[idx], self.hetero_graphs[idx]
	
	def __len__(self):
		return len(self.ids)	

	def extend(self , data):
		assert len(data.ids) == len(data.hetero_graphs)
		self.ids.extend(data.ids)
		self.hetero_graphs.extend(data.hetero_graphs)
		assert len(self.ids) == len(self.hetero_graphs)

def lig_mols_to_graph_multi_process(   lig ,explicit_H , use_chirality ,pl_c, seed_hg , ):
	try:
		hg = seed_hg.clone()
		mol_to_graph(lig, explicit_H=explicit_H, use_chirality=use_chirality, data=hg)
		create_hetero_graph(pl_c, hg)
	except Exception as e :
		traceback.print_exc()
		return None
	return hg

def get_ligname( m):
	if m is None:
		return None
	else:
		if m.HasProp("_Name"):
			return m.GetProp("_Name")
		else:
			return None

def _mol2_split(  infile):
	contents = open(infile, 'r').read()
	return ["@<TRIPOS>MOLECULE\n" + c for c in contents.split("@<TRIPOS>MOLECULE\n")[1:]]

def _sdf_split( infile):
	contents = open(infile, 'r').read()
	return [c + "$$$$\n" for c in contents.split("$$$$\n")[:-1]]

def load_prot(prot,explicit_H ,use_chirality):
	if isinstance(prot, Chem.rdchem.Mol):
		prot_mol = prot
	else:
		pocket = load_mol(prot, explicit_H=explicit_H, use_chirality=use_chirality)
		prot_mol = pocket
	return prot_mol

def load_ligs(ligs,explicit_H , use_chirality ,flag  ):
	if isinstance(ligs, np.ndarray) or isinstance(ligs, list):
		if isinstance(ligs[0], Chem.rdchem.Mol):
			ligs_mols = ligs
		else:
			raise ValueError('Ligands should be a list of rdkit.Chem.rdchem.Mol objects')
	else:
		if ligs.endswith(".mol2"):
			def tmp_process1(t_block):
				t_mol = Chem.MolFromMol2Block(t_block, removeHs=not explicit_H)
				if t_mol and use_chirality:
					Chem.AssignStereochemistryFrom3D(t_mol)
				return t_mol
			lig_blocks = _mol2_split(ligs)
			ligs_mols = [tmp_process1(lig_block) for lig_block in lig_blocks]

		elif ligs.endswith(".sdf"):
			def tmp_process2(t_block):
				t_mol = Chem.MolFromMolBlock(t_block, removeHs=not explicit_H)
				if t_mol and  use_chirality:
					Chem.AssignStereochemistryFrom3D(t_mol)
				return t_mol
			lig_blocks = _sdf_split(ligs)
			ligs_mols = [tmp_process2(lig_block) for lig_block in lig_blocks]

	if ligs_mols is not None:
		idsx = [  "%s-%s" % (get_ligname(lig), i)   if flag == '' else  "%s-%s-%s" % (  flag , get_ligname(lig), i) for i, lig in enumerate(ligs_mols)]
	else:
		raise ValueError('No ligands were given')
	return ligs_mols , idsx

def build_seed_hg(prot, cutoff, reflig, gen_pocket, explicit_H, use_chirality):
	prot_mol = load_prot(prot,explicit_H ,use_chirality)
	seed_hg = HeteroData()
	prot_to_graph(prot_mol, cutoff, seed_hg)
	return seed_hg
