import torch as th
import numpy as np
import random
import torch.nn.functional as F
from torch.distributions import Normal
from torch_scatter import scatter_add
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from scipy.stats import pearsonr, spearmanr

class Meter(object):
    def __init__(self, mean=None, std=None):
        self.mask = []
        self.y_pred = []
        self.y_true = []
        if (mean is not None) and (std is not None):
            self.mean = mean.cpu()
            self.std = std.cpu()
        else:
            self.mean = None
            self.std = None

    def update(self, y_pred, y_true, mask=None):
        self.y_pred.append(y_pred.detach().cpu())
        self.y_true.append(y_true.detach().cpu())
        if mask is None:
            self.mask.append(th.ones(self.y_pred[-1].shape))
        else:
            self.mask.append(mask.detach().cpu())

    def _finalize(self):
        mask = th.cat(self.mask, dim=0)
        y_pred = th.cat(self.y_pred, dim=0)
        y_true = th.cat(self.y_true, dim=0)

        if (self.mean is not None) and (self.std is not None):
            y_pred = y_pred * self.std + self.mean
        return mask, y_pred, y_true

    def _reduce_scores(self, scores, reduction='none'):
        if reduction == 'none':
            return scores
        elif reduction == 'mean':
            return np.mean(scores)
        elif reduction == 'sum':
            return np.sum(scores)
        else:
            raise ValueError(
                "Expect reduction to be 'none', 'mean' or 'sum', got {}".format(reduction))

    def multilabel_score(self, score_func, reduction='none'):
        mask, y_pred, y_true = self._finalize()
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0]
            task_y_pred = y_pred[:, task][task_w != 0]
            task_score = score_func(task_y_true, task_y_pred)
            if task_score is not None:
                scores.append(task_score)
        return self._reduce_scores(scores, reduction)

    def pearson_r(self, reduction='none'):
        def score(y_true, y_pred):
            return pearsonr(y_true.numpy(), y_pred.numpy())[0]
        return self.multilabel_score(score, reduction)

    def spearman_r(self, reduction='none'):
        def score(y_true, y_pred):
            return spearmanr(y_true.numpy(), y_pred.numpy())[0]
        return self.multilabel_score(score, reduction)

    def mae(self, reduction='none'):
        def score(y_true, y_pred):
            return F.l1_loss(y_true, y_pred).data.item()
        return self.multilabel_score(score, reduction)

    def rmse(self, reduction='none'):
        def score(y_true, y_pred):
            return th.sqrt(F.mse_loss(y_pred, y_true).cpu()).item()
        return self.multilabel_score(score, reduction)

    def roc_auc_score(self, reduction='none'):
        assert (self.mean is None) and (self.std is None), \
            'Label normalization should not be performed for binary classification.'
        def score(y_true, y_pred):
            if len(y_true.unique()) == 1:
                print('Warning: Only one class {} present in y_true for a task. '
                      'ROC AUC score is not defined in that case.'.format(y_true[0]))
                return None
            else:
                return roc_auc_score(y_true.long().numpy(), th.sigmoid(y_pred).numpy())
        return self.multilabel_score(score, reduction)

    def pr_auc_score(self, reduction='none'):
        assert (self.mean is None) and (self.std is None), \
            'Label normalization should not be performed for binary classification.'
        def score(y_true, y_pred):
            if len(y_true.unique()) == 1:
                print('Warning: Only one class {} present in y_true for a task. '
                      'PR AUC score is not defined in that case.'.format(y_true[0]))
                return None
            else:
                precision, recall, _ = precision_recall_curve(
                    y_true.long().numpy(), th.sigmoid(y_pred).numpy())
                return auc(recall, precision)
        return self.multilabel_score(score, reduction)

    def compute_metric(self, metric_name, reduction='none'):
        if metric_name == 'rp':
            return self.pearson_r(reduction)
        elif metric_name == 'rs':
            return self.spearman_r(reduction)
        elif metric_name == 'mae':
            return self.mae(reduction)
        elif metric_name == 'rmse':
            return self.rmse(reduction)
        elif metric_name == 'roc_auc_score':
            return self.roc_auc_score(reduction)
        elif metric_name == 'pr_auc_score':
            return self.pr_auc_score(reduction)
        elif metric_name == 'return_pred_true':
            return self.return_pred_true()
        else:
            raise ValueError('Expect metric_name to be "rp" or "rs" or "mae" or "rmse" '
                             'or "roc_auc_score" or "pr_auc", got {}'.format(metric_name))

class EarlyStopping(object):
    def __init__(self, mode='higher', patience=10, filename=None, metric=None):
        if filename is None:
            filename = 'early_stop.pth'
        if metric is not None:
            assert metric in ['rp', 'rs', 'mae', 'rmse', 'roc_auc_score', 'pr_auc_score'], \
                "Expect metric to be 'rp' or 'rs' or 'mae' or " \
                "'rmse' or 'roc_auc_score', got {}".format(metric)
            if metric in ['rp', 'rs', 'roc_auc_score', 'pr_auc_score']:
                print('For metric {}, the higher the better'.format(metric))
                mode = 'higher'
            if metric in ['mae', 'rmse']:
                print('For metric {}, the lower the better'.format(metric))
                mode = 'lower'

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = 0
        self.timestep = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False

    def _check_higher(self, score, prev_best_score):
        return score > prev_best_score

    def _check_lower(self, score, prev_best_score):
        return score < prev_best_score

    def step(self, score, model):
        self.timestep += 1
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        th.save({'model_state_dict': model.state_dict(),
                    'timestep': self.timestep}, self.filename)

    def load_checkpoint(self, model):
        model.load_state_dict(th.load(self.filename)['model_state_dict'])

def mdn_loss_fn(pi, sigma, mu, y, eps1=1e-10, eps2=1e-10):
    normal = Normal(mu, sigma)
    loglik = normal.log_prob(y.expand_as(normal.loc))
    prob = (th.log(pi + eps1) + loglik).exp().sum(1)
    loss = -th.log(prob + eps2)
    return loss, prob

def run_a_train_epoch(epoch, model, data_loader, optimizer, aux_weight=0.001 , affi_weight = 0., val_dist_threshold = 7., device='cpu'):
    idx =0
    model.train()
    total_loss = 0
    mdn_loss = 0
    atom_loss = 0
    bond_loss = 0
    affi_loss = 0
    batch_num =  len(data_loader)
    dist_in_threshold = 0
    dist_total = 0
    for batch_id, batch_data in enumerate(data_loader):
        idx+=1
        pdbids, b_hetero_g ,labels  = batch_data
        b_hetero_g = b_hetero_g.to(device)
        atom_labels = th.argmax( b_hetero_g['ligand'].feats[:,:17], dim=1, keepdim=False)
        bond_labels = th.argmax(  b_hetero_g['ligand','l2l' ,'ligand'].feats[:,:4], dim=1, keepdim=False)
        pi, sigma, mu, dist, atom_types, bond_types, batch = model( b_hetero_g )
        mdn, prob = mdn_loss_fn(pi, sigma, mu, dist)
        mdn = mdn[th.where(dist <= model.dist_threhold)[0]]
        dist_total += dist.shape[0]
        dist_in_threshold += mdn.shape[0]
        mdn = mdn.mean()
        batch = batch.to(device)
        if val_dist_threshold is not None:
            prob = prob[th.where(dist <= val_dist_threshold)[0]]
            y = scatter_add(prob, batch[th.where(dist <= val_dist_threshold)[0]], dim=0, dim_size=batch.unique().size(0))
        else:
            y = scatter_add(prob, batch, dim=0, dim_size=batch.unique().size(0))
        labels = labels.float().type_as(y).to(device)
        affi = th.corrcoef(th.stack([y, labels]))[1, 0]
        atom = F.cross_entropy(atom_types, atom_labels)
        bond = F.cross_entropy(bond_types, bond_labels)
        loss = mdn + (affi * affi_weight) + (atom * aux_weight) + (bond * aux_weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if idx%20 == 0:
            print('epoch{}:{}/{} total_loss:{} mdn_loss:{} atom_loss:{} bond_loss:{}'.format(epoch , idx
                                                                                     ,batch_num
                                                                                     ,loss.item()
                                                                                     ,mdn.item()
                                                                                     ,atom.item()
                                                                                     ,bond.item())
                  )
        mdn_loss += mdn.item() * b_hetero_g.batch_size
        atom_loss += atom.item() * b_hetero_g.batch_size
        bond_loss += bond.item() * b_hetero_g.batch_size
        total_loss +=  mdn_loss + atom_loss * aux_weight + bond_loss * aux_weight
        if np.isinf(mdn_loss) or np.isnan(mdn_loss):
            break
        del prob,y,b_hetero_g, atom_labels, bond_labels, pi, sigma, mu, dist, atom_types, bond_types, batch, mdn, atom, bond, loss
        th.cuda.empty_cache()
    print('训练过程中，dist in threshold 比例为：{}'.format(dist_in_threshold /  dist_total * 100 ))
    return total_loss / len(data_loader.dataset)+ affi_loss* affi_weight, mdn_loss / len(data_loader.dataset), affi_loss, atom_loss / len(data_loader.dataset), bond_loss / len(data_loader.dataset)

def run_an_eval_epoch(model, data_loader, pred=False, dist_threhold=5.,affi_weight=1.0, aux_weight=0.001, device='cpu'):
    model.eval()
    total_loss = 0
    mdn_loss = 0
    atom_loss = 0
    bond_loss = 0
    probs = []
    with th.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            if pred:
                pdbids, b_hetero_g = batch_data
            else:
                pdbids,  b_hetero_g, labels = batch_data
            b_hetero_g = b_hetero_g.to(device)
            atom_labels = th.argmax(b_hetero_g['ligand'].feats[:,:17], dim=1, keepdim=False)
            bond_labels = th.argmax(b_hetero_g['ligand','l2l','ligand'].feats[:,:4], dim=1, keepdim=False)
            pi, sigma, mu, dist, atom_types, bond_types, batch = model(b_hetero_g)
            if pred:
                prob = calculate_probablity(pi, sigma, mu, dist)
                prob[th.where(dist > dist_threhold)[0]] = 0.
                batch = batch.to(device)
                probx = scatter_add(prob, batch, dim=0, dim_size=b_hetero_g.batch_size)
                probs.append(probx)
            else:
                mdn, prob  = mdn_loss_fn(pi, sigma, mu, dist)
                mdn = mdn[th.where(dist <= model.dist_threhold)[0]]
                mdn = mdn.mean()
                batch = batch.to(device)
                prob = prob[th.where(dist <= dist_threhold)[0]]
                y = scatter_add(prob, batch[th.where(dist <= dist_threhold)[0]], dim=0,
                                dim_size=batch.unique().size(0))
                atom = F.cross_entropy(atom_types, atom_labels)
                bond = F.cross_entropy(bond_types, bond_labels)
                loss = mdn + (atom * aux_weight) + (bond * aux_weight)
                probs.append(y)
                total_loss += loss.item() * b_hetero_g.batch_size
                mdn_loss += mdn.item() * b_hetero_g.batch_size
                atom_loss += atom.item() * b_hetero_g.batch_size
                bond_loss += bond.item() * b_hetero_g.batch_size
            del b_hetero_g, atom_labels, bond_labels, pi, sigma, mu, dist, atom_types, bond_types, batch
            th.cuda.empty_cache()
    if pred:
        preds = th.cat(probs)
        return preds.cpu().detach().numpy()
    else:
        ys = th.cat(probs)
        affi_loss = th.corrcoef(th.stack([ys, data_loader.dataset.labels.to(device)]))[1, 0].item()
        del ys
        return total_loss / len(data_loader.dataset) + affi_loss * affi_weight , mdn_loss / len(data_loader.dataset), affi_loss, atom_loss / len(data_loader.dataset), bond_loss / len(data_loader.dataset)

def calculate_probablity(pi, sigma, mu, y):
    normal = Normal(mu, sigma)
    logprob = normal.log_prob(y.expand_as(normal.loc))
    logprob += th.log(pi)
    prob = logprob.exp().sum(1)
    return prob

def set_random_seed(seed=10):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)




