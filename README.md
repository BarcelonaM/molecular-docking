# InterFocusGT(InterFocusGCN)
InterFocusGT and InterFocusGCN are two models we designed based on an inter-edge-focused perspective—that a model should be centered explicitly on the protein–ligand interaction interface. 
They employ Graph Transformer and Gated Graph Convolutional Network, respectively, as intramolecular encoders, combined with dedicated interaction modules to aggregate cross-interface information.
The protein-ligand complex is first represented as a heterogeneous graph, where the outer green and inner blue subgraphs correspond to the protein and ligand graphs, respectively. 
The red region between them denotes the intermolecular interaction interface, with red nodes serving as inter-edges to represent residue-atom interactions. 
The protein graph, ligand graph, and interaction interface are fed into the node representation learning module (employing GT and MLAI for InterFocusGT, and GatedGCN and MLGCI for InterFocusGCN) for node representation learning. 
Subsequently, the learned node features are concatenated pairwise and then fed into the MDN to output the probability density distribution of residue-atom distances.
<img width="1361" height="1291" alt="image" src="https://github.com/user-attachments/assets/979ea20c-a6a3-4829-8bc5-b2c97a347c86" />
<img width="1853" height="1020" alt="image" src="https://github.com/user-attachments/assets/d92ab3c4-3a8f-4434-aaad-c3588bc6ae63" />

# Requirements
joblib==1.5.3
matplotlib==3.10.8
mda-xdrlib==0.2.0
MDAnalysis==2.10.0
numpy==2.3.5
pandas==3.0.0
rdkit==2025.9.4
scikit-learn==1.8.0
scipy==1.17.0
torch==2.7.1+cu128
torch-geometric==2.7.0
torch_scatter==2.1.2+pt27cu128
torchaudio==2.7.1+cu128
torchvision==0.22.1+cu128
tqdm==4.67.3
yarl==1.22.0

# Datasets
The PDBbind dataset and CASF-2016 benchmark are available at http://www.pdbbind.org.cn. 
The PDBbind-CrossDocked-Core dataset can be found at https://zenodo.org/record/5525936. 
The DUD-E and Dekois2.0 are available at https://dude.docking.org and http://www.dekois.com, respectively.
# Getting Started 
scripts/train_model.py  # training InterFocusGT
scripts/train_model_gcn.py # training InterFocusGCN
InterFocusGT/feats/mol2graph_rdmda_res.py # graph representation features encompass the construction of a protein graph, a ligand graph, and an interface graph.
saved_model/InterFocusGT.pth # trained model for InterFocusGT 


