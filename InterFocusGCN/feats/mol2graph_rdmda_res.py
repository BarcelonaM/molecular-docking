import numpy as np
from rdkit import Chem
import torch as th
import re, os
from itertools import permutations
from torch_geometric.data import HeteroData
from rdkit.Chem import SanitizeFlags
from scipy.spatial import distance_matrix
from joblib import Parallel, delayed
import MDAnalysis as mda
from MDAnalysis.analysis import distances
import traceback
from joblib import  dump

METAL = ["LI", "NA", "K", "RB", "CS", "MG", "TL", "CU", "AG", "BE", "NI", "PT", "ZN", "CO", "PD", "AG", "CR", "FE", "V",
         "MN", "HG", 'GA',
         "CD", "YB", "CA", "SN", "PB", "EU", "SR", "SM", "BA", "RA", "AL", "IN", "TL", "Y", "LA", "CE", "PR", "ND",
         "GD", "TB", "DY", "ER",
         "TM", "LU", "HF", "ZR", "CE", "U", "PU", "TH"]
RES_MAX_NATOMS = 100

def prot_to_graph(prot, cutoff , data ):
    u = mda.Universe(prot)
    res_feats = np.array([calc_res_features(res) for res in u.residues])
    data['protein'].feats = th.tensor(res_feats)
    data['protein'].num_nodes = data['protein'].feats.size(0)
    edgeids, distm = obatin_edge(u, cutoff)
    min_edgeids , _ = obatin_edge(u, 3)
    src_list, dst_list = zip(*edgeids)
    min_src_list , min_dst_list = zip(*min_edgeids)
    edge_index = th.tensor(  np.stack([src_list , dst_list] , axis=0 ))
    min_edge_index = th.tensor(  np.stack([min_src_list , min_dst_list] , axis=0 ))
    data['protein' , 'p2p'  , 'protein'].edge_index = edge_index
    data['protein' , 'p2p' , 'protein'].min_edge_index = min_edge_index
    ca_pos = th.tensor(np.array([obtain_ca_pos(res) for res in u.residues]))
    center_pos = th.tensor(u.atoms.center_of_mass(compound='residues'))
    dis_matx_ca = distance_matrix(ca_pos , ca_pos)
    cadist = th.tensor([dis_matx_ca[i, j] for i, j in edgeids]) * 0.1
    dis_matx_center = distance_matrix(center_pos , center_pos)
    cedist = th.tensor([dis_matx_center[i, j] for i, j in edgeids]) * 0.1
    edge_connect = th.tensor(np.array([check_connect(u, x, y) for x, y in zip(src_list, dst_list)]))
    data['protein' , 'p2p' , 'protein'].feats = th.cat([edge_connect.view(-1, 1), cadist.view(-1, 1), cedist.view(-1, 1), th.tensor(distm)],dim=1)
    data['protein'].pos = th.tensor(np.array([np.concatenate([res.atoms.positions, np.full((RES_MAX_NATOMS - len(res.atoms), 3), np.nan)], axis=0) for res in u.residues]))  # 残基的每个原子的pos都记录下来

def obtain_ca_pos(res):
    if obtain_resname(res) == "M":
        return res.atoms.positions[0]
    else:
        try:
            pos = res.atoms.select_atoms("name CA").positions[0]
            return pos
        except:
            return res.atoms.positions.mean(axis=0)

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def obtain_self_dist(res):
    try:
        xx = res.atoms
        dists = distances.self_distance_array(xx.positions)
        ca = xx.select_atoms("name CA")
        c = xx.select_atoms("name C")
        n = xx.select_atoms("name N")
        o = xx.select_atoms("name O")
        return [dists.max() * 0.1, dists.min() * 0.1, distances.dist(ca, o)[-1][0] * 0.1,
                distances.dist(o, n)[-1][0] * 0.1, distances.dist(n, c)[-1][0] * 0.1]
    except:
        return [0, 0, 0, 0, 0]

def obtain_dihediral_angles(res):
    try:
        if res.phi_selection() is not None:
            phi = res.phi_selection().dihedral.value()
        else:
            phi = 0
        if res.psi_selection() is not None:
            psi = res.psi_selection().dihedral.value()
        else:
            psi = 0
        if res.omega_selection() is not None:
            omega = res.omega_selection().dihedral.value()
        else:
            omega = 0
        if res.chi1_selection() is not None:
            chi1 = res.chi1_selection().dihedral.value()
        else:
            chi1 = 0
        return [phi * 0.01, psi * 0.01, omega * 0.01, chi1 * 0.01]
    except:
        return [0, 0, 0, 0]

def calc_res_features(res):
    return np.array(one_of_k_encoding_unk(obtain_resname(res),
                                          ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR',
                                           'TRP', 'SER', 'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP',
                                           'GLU', 'LYS', 'ARG', 'HIS', 'MSE', 'CSO', 'PTR', 'TPO',
                                           'KCX', 'CSD', 'SEP', 'MLY', 'PCA', 'LLP', 'M', 'X']) +  # 32  residue type
                    obtain_self_dist(res) +  # 5
                    obtain_dihediral_angles(res)  # 4
                    )

def obtain_resname(res):
    if res.resname[:2] == "CA":
        resname = "CA"
    elif res.resname[:2] == "FE":
        resname = "FE"
    elif res.resname[:2] == "CU":
        resname = "CU"
    else:
        resname = res.resname.strip()
    if resname in METAL:
        return "M"
    else:
        return resname

def obatin_edge(u, cutoff=10.0):
    edgeids = []
    dismin = []
    dismax = []
    for res1, res2 in permutations(u.residues, 2):
        dist = calc_dist(res1, res2)
        if dist.min() <= cutoff:
            edgeids.append([res1.ix, res2.ix])
            dismin.append(dist.min() * 0.1)
            dismax.append(dist.max() * 0.1)
    return edgeids, np.array([dismin, dismax]).T

def check_connect(u, i, j):
    if abs(i - j) != 1:
        return 0
    else:
        if i > j:
            i = j
        nb1 = len(u.residues[i].get_connections("bonds"))
        nb2 = len(u.residues[i + 1].get_connections("bonds"))
        nb3 = len(u.residues[i:i + 2].get_connections("bonds"))

        if nb1 + nb2 == nb3 + 1:
            return 1
        else:
            return 0

def calc_dist(res1, res2):
    dist_array = distances.distance_array(res1.atoms.positions, res2.atoms.positions)
    return dist_array

def calc_atom_features(atom, explicit_H=False):
    results = one_of_k_encoding_unk(
        atom.GetSymbol(),
        [
            'C', 'N', 'O', 'S', 'F', 'P', 'Cl',
            'Br', 'I', 'B', 'Si', 'Fe', 'Zn',
            'Cu', 'Mn', 'Mo', 'other'
        ]) + one_of_k_encoding(atom.GetDegree(),
                               [0, 1, 2, 3, 4, 5, 6]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                  Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                  Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                  Chem.rdchem.HybridizationType.SP3D2, 'other']) + [atom.GetIsAromatic()]
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),  #
                                                  [0, 1, 2, 3, 4])
    return np.array(results)

def calc_bond_features(bond, use_chirality=True):
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    return np.array(bond_feats).astype(int)

def load_mol(molpath, explicit_H=False, use_chirality=True):
    if re.search(r'.pdb$', molpath):
        mol = Chem.MolFromPDBFile(molpath, removeHs=not explicit_H)

    elif re.search(r'.mol2$', molpath):
        mol = Chem.MolFromMol2File(molpath, removeHs=not explicit_H)

    elif re.search(r'.sdf$', molpath):
        mol = Chem.MolFromMolFile(molpath, removeHs=not explicit_H, sanitize=False)
        mol = ligand_process_without_kekulize(mol,explicit_H)
    else:
        raise IOError("only the molecule files with .pdb|.sdf|.mol2 are supported!")

    if use_chirality:
        Chem.AssignStereochemistryFrom3D(mol)
    return mol

def ligand_process_without_kekulize(mol,eH):
    if not eH:
        mol = Chem.RemoveHs(mol, implicitOnly=False, updateExplicitCount=False, sanitize=False)
    Chem.SanitizeMol(mol, sanitizeOps=SanitizeFlags.SANITIZE_ALL ^ SanitizeFlags.SANITIZE_KEKULIZE)
    return mol

def mol_to_graph(mol, explicit_H=False, use_chirality=True , data=None ):
    num_atoms = mol.GetNumAtoms()
    atom_feats = np.array([calc_atom_features(a, explicit_H=explicit_H) for a in mol.GetAtoms()])
    if use_chirality:
        chiralcenters = Chem.FindMolChiralCenters(Chem.Mol(mol.ToBinary()), force=True, includeUnassigned=True,
                                                  useLegacyImplementation=False)
        chiral_arr = np.zeros([num_atoms, 3])
        for (i, rs) in chiralcenters:
            if rs == 'R':
                chiral_arr[i, 0] = 1
            elif rs == 'S':
                chiral_arr[i, 1] = 1
            else:
                chiral_arr[i, 2] = 1
        atom_feats = np.concatenate([atom_feats, chiral_arr], axis=1)
    data['ligand'].feats = th.tensor(atom_feats)
    data['ligand'].num_nodes = data['ligand'].feats.size(0)
    atomCoords = mol.GetConformer().GetPositions()
    data['ligand'].pos = th.tensor(atomCoords)
    src_list = []
    dst_list = []
    bond_feats_all = []
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        bond_feats = calc_bond_features(bond, use_chirality=use_chirality)
        src_list.extend([u, v])
        dst_list.extend([v, u])
        bond_feats_all.append(bond_feats)
        bond_feats_all.append(bond_feats)

    edge_index = th.tensor(np.stack([src_list, dst_list], axis=0))
    data['ligand' , 'l2l' , 'ligand'].edge_index = edge_index
    data['ligand' , 'l2l' , 'ligand'].feats = th.tensor(np.array(bond_feats_all))

def get_p2l_dist(p , l, p_node_pos , l_node_pos ):
    pos_p  = p_node_pos[p]
    pos_l = l_node_pos[l]
    pos_p = pos_p.double()
    pos_l = pos_l.double().view(-1,  3)
    cha_ = pos_p - pos_l
    d = th.nan_to_num( th.sum(cha_ ** 2 , axis = -1)  ** 0.5   ,  10000) .min()
    return d

def build_one_jump_target(edge_index):
    list1 = edge_index[0].tolist()
    list2 = edge_index[1].tolist()
    one_jump = {}
    for n1 ,n2 in zip(list1 ,list2):
        if n1 not in one_jump:
            one_jump[n1] = []
        one_jump[n1].append(n2)
    return one_jump

def create_hetero_graph(pl_c ,data ):
    p_node_feat = data['protein'].feats
    p_node_num = data['protein'].feats.shape[0]
    p_node_pos = data['protein'].pos
    l_node_feat = data['ligand'].feats
    l_node_num = data['ligand'].feats.shape[0]
    l_node_pos = data['ligand'].pos
    p_dikaer = th.arange(p_node_num)
    l_dikaer = th.arange(l_node_num)
    src = p_dikaer.repeat_interleave(l_node_num)
    dst = l_dikaer.repeat(p_node_num)
    p2l_edges = list(zip(src.tolist(), dst.tolist()))
    p2e_edge_index = []
    l2e_edge_index = []
    p2e_edge_feat = []
    l2e_edge_feat = []
    e_feats = []
    e_src_dst = []
    for  i , (p , l)  in  enumerate( p2l_edges):
        p_f = p_node_feat[p]
        l_f = l_node_feat[l]
        e_feats.append( th.concatenate([p_f , l_f ] ,dim  = -1) )
        e_src_dst.append((p,l))
        dist = get_p2l_dist(p, l, p_node_pos, l_node_pos)
        if dist > pl_c:
            continue
        p2e_edge_index.append((p,i))
        l2e_edge_index.append((l,i))
        p2e_edge_feat.append((1,0))
        l2e_edge_feat.append((1,0))

    if len(p2e_edge_index) == 0 :
        p2e_edge_index.append((0, 0))
        l2e_edge_index.append((0, 0))
        p2e_edge_feat.append((0, 0))
        l2e_edge_feat.append((0, 0))

    e_feats = th.stack(e_feats , dim = 0)
    e_src_dst = th.tensor(np.array(e_src_dst))
    p2e_edge_index = th.permute( th.tensor(np.array(p2e_edge_index)) , dims=[1,0]) # 2 , edge num
    l2e_edge_index = th.permute( th.tensor(np.array(l2e_edge_index)) , dims=[1,0]) # 2 , edge num
    p2e_edge_feat = th.tensor(np.array( p2e_edge_feat ))
    l2e_edge_feat = th.tensor(np.array( l2e_edge_feat ))
    data['inter_edge'].feats = e_feats
    data['inter_edge'].num_nodes =  e_feats.size(0)
    data['protein' ,'p2e' , 'inter_edge'].edge_index = p2e_edge_index
    data['ligand' , 'l2e' , 'inter_edge'].edge_index = l2e_edge_index
    data['protein' ,'p2e' , 'inter_edge'].feats = p2e_edge_feat
    data['ligand' , 'l2e' , 'inter_edge'].feats = l2e_edge_feat
    data['inter_edge'].src_dst = e_src_dst
    del data['protein' ,'p2p' ,'protein'].min_edge_index

def mol_to_graph2(prot_path, lig_path, cutoff=10.0, explicit_H=False, use_chirality=True , pl_c = 7):
    prot = load_mol(prot_path, explicit_H=explicit_H, use_chirality=use_chirality)
    lig = load_mol(lig_path, explicit_H=explicit_H, use_chirality=use_chirality)
    hetero_g = HeteroData()
    prot_to_graph(prot, cutoff , hetero_g)
    mol_to_graph(lig, explicit_H=explicit_H, use_chirality=use_chirality , data=hetero_g )
    create_hetero_graph(pl_c  ,hetero_g )
    return hetero_g


def pdbbind_handle(idx, pdbid, args):
    prot_path = "%s/%s/%s_pocket_10_preprocessed_1124.pdb" % (args.dir, pdbid, pdbid) # 1a28_self_pocket_preprocessed.pdb # 1a1e_protein_pocket_10.pdb
    lig_path = "%s/%s/%s_ligand.mol2" % (args.dir, pdbid, pdbid)
    try:
        hetero_g = mol_to_graph2(prot_path,
                               lig_path,
                               cutoff=args.cutoff,
                               explicit_H=args.useH,
                               use_chirality=args.use_chirality,
                               pl_c = args.pl_c)
    except Exception as e:
        traceback.print_exc()
        with open("%s\%s_out.log" % (args.out_dir,args.out_prefix), 'a') as f:
            f.write('failed_pdb_id {}\n'.format(pdbid))
        print("%s failed to generare the graph" % pdbid)
        hetero_g =   None

    return pdbid, hetero_g , label_query(pdbid, args.ref)

def label_query(pdbid, maps):
    return  maps[pdbid]

def UserInput():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--dir', default=".",
                   help='The directory to store the protein-ligand complexes.')
    p.add_argument('-c', '--cutoff', default=10, type=float,
                   help='the cutoff to determine the pocket')
    p.add_argument('-pl_c' , '--pl_c', default=7, type=float)
    p.add_argument('-out_dir', '--out_dir', default="",
                   help='The output bin file.')
    p.add_argument('-out_prefix', '--out_prefix', default="",
                   help='')
    p.add_argument('-usH', '--useH', default=False, action="store_true",
                   help='whether to use the explicit H atoms.')
    p.add_argument('-uschi', '--use_chirality', default=False, action="store_true",
                   help='whether to use chirality.')
    p.add_argument('-ref', '--ref'  , default="data/PDBbind_v2020_plain_text_index/index/INDEX_general_PL_data.2020",)
    p.add_argument('-p', '--parallel', default=False, action="store_true",
                   help='whether to obtain the graphs in parallel (When the dataset is too large,\
						 it may be out of memory when conducting the parallel mode).')
    args = p.parse_args()
    return args

def main():
    args = UserInput()
    pdbids = [x for x in os.listdir(args.dir) if
              os.path.isdir("%s/%s" % (args.dir, x)) and x != 'index' and x != 'readme']
    affinity_dict = {}
    with open(args.ref, "r") as f:
        for line in f:
            pdb_id = line.split()[0]
            affinity_score = float(line.split()[3])
            affinity_dict[pdb_id] = affinity_score
    args.ref = affinity_dict

    if args.parallel:
        results = Parallel(n_jobs=-1)(delayed(pdbbind_handle)(idx, pdbid, args) for idx, pdbid in enumerate(pdbids))
    else:
        results = []
        for idx, pdbid in enumerate(pdbids):
            results.append(pdbbind_handle(idx, pdbid, args))

    results = list(filter(lambda x: x[1] != None, results))
    ids, hetero_g ,labels = list(zip(*results))
    np.save("{}\\{}_idsresz.npy".format(args.out_dir ,args.out_prefix ) , (ids, labels))
    dump(list(hetero_g), "{}\\{}_hresz.bin".format(args.out_dir ,args.out_prefix))

if __name__ == '__main__':
    main()
