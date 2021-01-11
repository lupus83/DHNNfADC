from collections import defaultdict
from typing import Any, Dict, Iterable, Optional, Tuple
from sklearn.metrics import accuracy_score, balanced_accuracy_score, average_precision_score, roc_auc_score, confusion_matrix
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models import ModelFactory, ResBlock
from utils import AdniVolumeDataset, load_heterogeneous_data, split_data, make_confusion_matrix
import os
import pandas as pd
import copy


LOG = logging.getLogger(__name__)

def train_and_evaluate(model: nn.Module,
                       model_dir: Path,
                       train_loader: DataLoader,
                       eval_loader: DataLoader,
                       n_epochs: int,
                       lr: float,
                       optimizer: str='Adam',
                       betas: Tuple[float, float]=(0.9, 0.999),
                       weight_decay: float=0,
                       hnn: bool=False,
                       transer_learning_exp: bool=False,
                       hyperparams: Dict[Any, Any]={}):
    """ fit() function for this project

    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Generally: before constructing optimizers, make sure model is already on GPU!
    model = model.to(device)
    trainer = Trainer(model, train_loader, optimizer, lr=lr, betas=betas, weight_decay=weight_decay, device=device, n_epochs=n_epochs, transer_learning_exp=transer_learning_exp)
    evaluator = Evaluator(model, eval_loader, device=device, hnn=hnn)

    train_summary = SummaryWriter(str(model_dir / "train"))
    eval_summary = SummaryWriter(str(model_dir / "val"))

    model_dir = model_dir / "checkpoints"
    try:
        model_dir.mkdir()
    except FileExistsError:
        print("")

    # initialize in order to tune hyperparams
    best_model_acc = -1
    best_model_bacc = -1
    best_model_loss = 1000

    pbar = tqdm(range(n_epochs))
    for epoch in pbar:
        if hnn:
            stats = trainer.train_hnn_one_epoch(pbar)
        else:
            stats = trainer.train_one_epoch(pbar)

        pbar.set_postfix(stats)

        for key, value in stats.items():
            train_summary.add_scalar(key, value, epoch)


        for tag, parm in model.named_parameters():
            if parm.grad is not None:
                train_summary.add_histogram(tag, parm.grad.data.cpu().numpy(), epoch)


        eval_stats = evaluator.evaluate()
        for key, value in eval_stats.items():
            eval_summary.add_scalar(key, value, epoch)

        # save model if improvement. Format to 2 decimals
        if eval_stats['balanced_accuracy'] > best_model_bacc:
            best_model_acc = eval_stats['accuracy']
            best_model_bacc = eval_stats['balanced_accuracy']
            best_model_loss = eval_stats['loss']
            model_file = model_dir / \
                '{}_epoch_{:02d}.pt'.format(model.__class__.__name__, epoch)
            LOG.debug("Saving model with balanced accuracy %.2f to %s", best_model_bacc, model_file)
            torch.save(model.state_dict(), model_file)
            torch.save(model.state_dict(), (model_dir / 'Best.pt'))


    # save hyperparams
    model_dir_str = str(model_dir)
    vol = model_dir_str[model_dir_str.find("VolMask")+8:]
    if (vol.startswith("concat")):
        vol = "concat"
    elif (vol.startswith("vol_with_bg")):
        vol = "vol_with_bg"
    elif (vol.startswith("vol_without_bg")):
        vol = "vol_without_bg"
    elif (vol.startswith("mask")):
        vol = "mask"

    seed = model_dir_str[model_dir_str.find("Seed")+5:]
    seed = seed[:seed.find("_")]

    opti = str(trainer._optimizer)
    opti = opti[:opti.index(' ')]

    hypers = copy.deepcopy(hyperparams)
    hypers['n_epochs'] = str(n_epochs)
    hypers['lr'] = str(lr)
    hypers['weight_decay'] = str(weight_decay)
    hypers['batch_size'] = str(eval_loader.batch_size)
    hypers['volume'] = vol
    hypers['seed'] = seed
    hypers['optimizer'] = opti

    if trainer._scheduler is not None:
        sched = str(trainer._scheduler)
        if 'Trapezoid' in sched:
            sched = 'Trapezoid'
        else:
            sched = sched[sched.index('lr_scheduler.')+13:]
            sched = sched[:sched.index(' ')]
        hypers['scheduler'] = sched

    scores = {'accuracy': best_model_acc,
        'loss': best_model_loss,
        'balanced_accuracy': best_model_bacc}
    eval_summary.add_hparams(hypers, scores)

    train_summary.flush()
    train_summary.close()
    eval_summary.flush()
    eval_summary.close()


class Trainer:
    """ Trainer class
        
    """
    def __init__(self,
                 model: nn.Module,
                 dataloader: DataLoader,
                 optimizer: str,
                 lr: float = 0.05,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 weight_decay: float = 0.0,
                 device: Any = None,
                 n_epochs: int = 50,
                 verbose: bool = True,
                 transer_learning_exp: bool = False) -> None:
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.is_transfer = transer_learning_exp

        # Generally: before constructing optimizers, make sure model is already on GPU!
        if optimizer == 'AdamW':
            self._optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, betas=betas, weight_decay=weight_decay)
            self._scheduler = Trapezoid(self._optimizer, n_epochs*len(dataloader), lr)
            # self._scheduler = None
        elif optimizer == 'Adam':  # just used for 'plain' experimenting
            self._optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()))
            self._scheduler = None
        elif optimizer == 'SGD':
            self._optimizer =  optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, momentum=0.9)
            self._scheduler = Trapezoid(self._optimizer, n_epochs*len(dataloader), lr)
            # self._scheduler = None
            # self._scheduler = optim.lr_scheduler.OneCycleLR(self._optimizer,
            #                                                 max_lr=lr,
            #                                                 total_steps=n_epochs*len(dataloader),
            #                                                 pct_start=0.2,
            #                                                 anneal_strategy='linear',
            #                                                 cycle_momentum=True)

        else:
            raise ValueError(optimizer)


    def get_current_lr(self):

        for pg in self._optimizer.param_groups:
            return pg['lr']

    def train_one_epoch(self, pbar: Optional[tqdm] = None) -> Dict[str, float]:
        device = self.device

        model = self.model
        model.train()
        if self.is_transfer:
            model.apply(deactivate_runningstats_bn)

        train_loss = 0.0
        correct = 0
        total = 0
        num_samples = 0
        for batch_idx, data in enumerate(self.dataloader):

            if (len(data) == 2):  # non heterogeneous network
                vols, label = data[0].to(device), data[1].to(device)
                prediction = model(vols)
            elif (len(data) == 3):  # heterogeneous network
                vols, non_image_data, label = data[0].to(device), data[1].to(device), data[2].to(device)
                prediction = model(vols, non_image_data)
            else:
                raise ValueError(f'Invalid batch at index {batch_idx} with len {len(data)}')

            is_multiclass = prediction.shape[1] > 1
            if is_multiclass:
                criterion = nn.CrossEntropyLoss()
                label = torch.squeeze(label, 1)
                label = label.long()
                _, pred_class = torch.max(prediction.data, 1)
            else:
                criterion = nn.BCEWithLogitsLoss()
                prediction_prob = torch.sigmoid(prediction)
                pred_class = (prediction_prob.data > 0.5).type(torch.FloatTensor).to(device)
            loss = criterion(prediction, label)

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            if self._scheduler is not None:
                self._scheduler.step()

            train_loss += loss.item()
            total += 1
            num_samples += label.size(0)
            correct += (pred_class == label).sum().item()

        if pbar is not None:
            pbar.update()

        acc = correct / num_samples
        train_loss /= total
        return {'loss': train_loss, 'accuracy': acc}

    def train_hnn_one_epoch(self, pbar: Optional[tqdm] = None) -> Dict[str, float]:
        device = self.device

        model = self.model
        model.train()
        if self.is_transfer:
            model.apply(deactivate_runningstats_bn)

        train_loss = 0.0
        correct = 0
        total = 0
        num_samples = 0
        for batch_idx, (vols, non_image_data, label) in enumerate(self.dataloader):
            vols, non_image_data, label = vols.to(device), non_image_data.to(device), label.to(device)
            prediction = model(vols, non_image_data)
            is_multiclass = prediction.shape[1] > 1
            if is_multiclass:
                criterion = nn.CrossEntropyLoss()
                label = torch.squeeze(label, 1)
                label = label.long()
                _, pred_class = torch.max(prediction.data, 1)
            else:
                criterion = nn.BCEWithLogitsLoss()
                prediction_prob = torch.sigmoid(prediction)
                pred_class = (prediction_prob.data > 0.5).type(torch.FloatTensor).to(device)
            loss = criterion(prediction, label)

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            if self._scheduler is not None:
                self._scheduler.step()

            train_loss += loss.item()
            total += 1
            num_samples += label.size(0)
            correct += (pred_class == label).sum().item()

        if pbar is not None:
            pbar.update()

        acc = correct / num_samples
        train_loss /= total
        return {'loss': train_loss, 'accuracy': acc}

class Predictor:

    def __init__(self,
                 model: nn.Module,
                 dataloader: DataLoader,
                 device: Any = None,
                 with_label: bool = True) -> None:
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.with_label = with_label

    def predict(self) -> Iterable[Dict[str, torch.FloatTensor]]:
        device = self.device

        model = self.model
        model.eval()

        with torch.no_grad():
            for batch in self.dataloader:
                batch = [v.to(device) for v in batch]
                in_batch = batch[:-1] if self.with_label else batch
                prediction = model(*in_batch)
                is_multiclass = prediction.shape[1] > 1
                if is_multiclass:
                    prediction_prob = F.softmax(prediction, dim=1)
                    _, pred_class = torch.max(prediction.data, dim=1, keepdim=True)
                    # pred_class = pred_class.unsqueeze(1) -> no need if keepdim=True above
                else:
                    prediction_prob = torch.sigmoid(prediction)
                    pred_class = (prediction_prob.data > 0.5).type(torch.FloatTensor).to(device)

                data = {"logits": prediction,
                        "probabilities": prediction_prob,
                        "classes": pred_class}
                if self.with_label:
                    data["label"] = batch[-1]
                yield data

    def predict_numpy(self) -> Iterable[Dict[str, np.ndarray]]:
        for pred in self.predict():
            yield {k: v.cpu().numpy() for k, v in pred.items()}


class Evaluator(Predictor):
    def __init__(self,
                 model: nn.Module,
                 dataloader: DataLoader,
                 device: Any = None,
                 hnn: bool = False) -> None:
        super().__init__(model=model,
                         dataloader=dataloader,
                         device=device,
                         with_label=True)

    def evaluate(self) -> Dict[str, float]:
        test_loss = 0.0
        total = 0
        outputs = defaultdict(list)
        is_multiclass = False
        
        for pred in self.predict():
            for k, v in pred.items():
                outputs[k].append(v.cpu().detach().numpy())

            label = pred["label"]
            prediction = pred["logits"]
            is_multiclass = prediction.shape[1] > 1
            if is_multiclass:
                label = torch.squeeze(label, 1)
                label = label.long()
                criterion = nn.CrossEntropyLoss()
            else:
                criterion = nn.BCEWithLogitsLoss()
            loss = criterion(prediction, label)

            test_loss += loss.item()
            total += 1

        test_loss /= total

        pred_arrays = {}
        for k, v in outputs.items():
            
            last_dim_size = v[0].shape[-1]
            if last_dim_size == 1:  # check if last dimension has to be removed (exception -> multiclass logits and probabilities)
                v = [np.squeeze(vv, axis=last_dim_size) for vv in v]

            pred_arrays[k] = np.concatenate(v, axis=0)  # concat all batches

        metrics = {'loss': test_loss,
                   'accuracy': accuracy_score(y_true=pred_arrays["label"], y_pred=pred_arrays["classes"]),
                   'balanced_accuracy': balanced_accuracy_score(y_true=pred_arrays["label"], y_pred=pred_arrays["classes"])}
        if not is_multiclass:
            metrics["roc_auc"] = roc_auc_score(
                    y_true=pred_arrays["label"], y_score=pred_arrays["probabilities"])
            metrics["average_precision"] = average_precision_score(
                    y_true=pred_arrays["label"], y_score=pred_arrays["probabilities"])

        return metrics

def xavier_uniform_init(module: nn.Module):
    if isinstance(module, torch.nn.Conv3d) or isinstance(module, torch.nn.Linear):
        nn.init.xavier_uniform_(module.weight)


def batchnorm_init_zeros(module: nn.Module):
    """ found in https://github.com/aramis-lab/AD-DL. Adapted.
        Zero-initialize the last BN in each residual branch,
        so that the residual branch starts with zeros, and each residual block behaves like an identity.
        This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    """
    if isinstance(module, ResBlock):
        nn.init.constant_(module.bn2.weight, 0)


def deactivate_runningstats_bn(module: nn.Module):
    """ Use to properly freeze network when doing transfer learning
        set after every model.eval() amd model.train():
        function call "model.apply(deactivate_batchnorm)"
        Thanks und shoutouts an github user: @KilsenP
    """
    if isinstance(module, torch.nn.BatchNorm3d):
        if module.weight.requires_grad == False and module.bias.requires_grad == False:
            module.eval()

def deactivate_batchnorm(module: nn.Module):
    """ Used for overfitting a single batch
        set after every model.eval() amd model.train():
        function call "model.apply(deactivate_batchnorm)"
    """
    if isinstance(module, torch.nn.BatchNorm3d):
        module.reset_parameters()
        module.eval()
        with torch.no_grad():
            module.weight.fill_(1.0)
            module.bias.zero_()


class EarlyStopping(object):
    """ Lazy implementation of Early Stopping
        if no early stopping happened within
        patience iterations/epochs (depends on how step() is called)
        stop training
    """
    def __init__(self, patience, mode='loss'):

        self.patience = patience
        self.counter = 0
        self.best = None
        
        if mode == 'loss':
            self.is_better = lambda x, y: x < y
        else:
            self.is_better = lambda x, y: x > y

    def step(self, metric):

        if (torch.isnan(metric) == True and self.best == None):
            raise ValueError('Metric in first step diverged!')

        if torch.isnan(metric):
            return True

        if self.is_better(metric, self.best):
            self.counter = 0
            self.best = metric
        else:
            self.counter += 1

        if self.counter >= self.patience:
            return True

        return False

class Superconvergence(optim.lr_scheduler._LRScheduler):

    def __init__(self,
                optimizer,
                max_lr,
                epoch_annihilation: int,
                start_lr_fraction: int=50,
                last_epoch: int=-1):
        self.max_lr = max_lr
        self.init_lr = max_lr / start_lr_fraction
        x_steps = epoch_annihilation / 2
        self.epoch_annihilation = epoch_annihilation
        self.lr_step = (self.max_lr - self.init_lr) / x_steps
        super(Superconvergence, self).__init__(optimizer, last_epoch)
        """
        like torch.optim.lr_scheduler.OneCycleLR, but extended by an annihilaiton phase
        """

    def get_lr(self):

        if self.last_epoch >= self.epoch_annihilation:
            new_lr = self.init_lr - ((self.last_epoch - self.epoch_annihilation) * 0.01 * self.lr_step)
            new_lr = (new_lr if new_lr > 1e-8 else 1e-8)
        elif self.last_epoch < (self.epoch_annihilation / 2):
            new_lr =  self.init_lr + (self.last_epoch * self.lr_step)
        else:
            new_lr =  self.max_lr - (self.last_epoch - ((self.epoch_annihilation / 2) * self.lr_step))
        return [new_lr for group in self.optimizer.param_groups]

class Trapezoid(optim.lr_scheduler._LRScheduler):

    def __init__(self,
                optimizer,
                n_iterations: int,
                max_lr: float,
                start_lr: Optional[float]=None,
                annihilate: bool=True,
                last_epoch: int=-1
                ):
        """
            Lazy: n_iterations is the total amount of iterations that this scheduler will be used for!
            Developer's note:
            if cyclic momentum would be implemented, according to Superconvergence paper
            https://arxiv.org/abs/1708.07120
            0.85 as min val works just fine. Take that value!
        """
        

        self.n_iters = n_iterations
        self.max_lr = max_lr
        if start_lr is None:
            self.start_lr = max_lr / 10
        else:
            self.start_lr = start_lr
        self.stop_warmup = int(0.2 * n_iterations)
        self.start_decline = int(0.6 * n_iterations)
        self.start_annihilate = int(0.9 * n_iterations) if annihilate else n_iterations

        super(Trapezoid, self).__init__(optimizer, last_epoch)

    def get_lr(self):

        if self.last_epoch < self.stop_warmup:
            step_size = (self.max_lr - self.start_lr) / self.stop_warmup
            new_lr = self.start_lr + step_size * self.last_epoch
        elif self.last_epoch < self.start_decline:
            new_lr = self.max_lr
        elif self.last_epoch <= self.start_annihilate:
            step_size = (self.max_lr - self.start_lr) / (self.start_annihilate - self.start_decline)
            new_lr = self.max_lr - step_size * (self.last_epoch - self.start_decline)
        else:
            step_size = (self.start_lr - self.start_lr / 20) / (self.n_iters - self.start_annihilate)
            new_lr = self.start_lr - step_size * (self.last_epoch - self.start_annihilate)
            
        return [new_lr for group in self.optimizer.param_groups]

def predict_single_net(model, data_loader, gpu=0, is_binary=True) -> Dict[str, float]:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(gpu)

    model = model.to(device)

    predictor = Predictor(model, data_loader, device=device)

    outputs = defaultdict(list)

    for pred in predictor.predict():

        for k, v in pred.items():
            outputs[k].append(v.cpu().detach().numpy())

    pred_arrays = {}
    for k, v in outputs.items():

        last_dim_size = v[0].shape[-1]
        if last_dim_size == 1:
            v = [np.squeeze(vv, axis=last_dim_size) for vv in v]

        pred_arrays[k] = np.concatenate(v, axis=0)  # concat all batches

    metrics = {
           'accuracy': accuracy_score(y_true=pred_arrays["label"], y_pred=pred_arrays["classes"]),
           'balanced_accuracy': balanced_accuracy_score(y_true=pred_arrays["label"], y_pred=pred_arrays["classes"]),
           'confusion_matrix': confusion_matrix(y_true=pred_arrays["label"], y_pred=pred_arrays["classes"])}
    if is_binary:
            metrics["roc_auc"] = roc_auc_score(
                    y_true=pred_arrays["label"], y_score=pred_arrays["probabilities"])
            metrics["average_precision"] = average_precision_score(
                    y_true=pred_arrays["label"], y_score=pred_arrays["probabilities"])

    return metrics

def predict_net_on_adni(args, gpu=0):
    """
        args must contain:
        Paths:
        - model_path
        - path_to_image_data
        - path_to_non_image_data
        Data:
        - mask
        - hippocampus
        - seeds
        - filter_by_visitcodes
        - test_size
        Model:
        - model_type
        - input_dim
        - n_outputs
        - dependent: n_hidden_layer_aux
        - dependent: reduction_ratio
        Hyperparams:
        - bn_momentum
        - batch_size
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(gpu)
    net = args['model_type']
    
    is_se_net = True if ('SE' in net) else False

    heterogeneous_exp = True if (net in ModelFactory().get_heterogeneous_models()) else False
    batch_size = args['batch_size']
    in_channels = 2 if args['mask'] == 'concat' else 1

    if args['filter_by_visitcodes']:
        data, patients, vscodes = load_heterogeneous_data(args['path_to_image_data'], args['path_to_non_image_data'], \
                                                hippocampus=args['hippocampus'], masking=args['mask'], debug=False, \
                                                visit_codes=True)
        non_image_data_ndim = len(data[0][1])
    else:
        data, patients = load_heterogeneous_data(args['path_to_image_data'], args['path_to_non_image_data'], \
                                                hippocampus=args['hippocampus'], masking=args['mask'], debug=False, \
                                                visit_codes=False)

    df = []
    conf_matrix = [[], []]
    for seed in args['seeds']:

        splits = split_data(data, groups=patients, visit_codes=vscodes, test_size=args['test_size'], seed=seed)
        
        if heterogeneous_exp:
            ds = AdniVolumeDataset(splits["test"], dim=args['input_dim'], masking=args['mask'], predicting=True, heterog_exp=True)
            params = {'in_channels': in_channels, 'n_outputs': args['n_outputs'], 'bn_momentum': args['bn_momentum'], 'ndim_non_img': non_image_data_ndim}
            if net == 'DynamicHNN':
                params['n_hidden_layers_aux'] = args['n_hidden_layer_aux']
        else:
            ds = AdniVolumeDataset(splits["test"], dim=args['input_dim'], masking=args['mask'], predicting=True)
            params = {'in_channels': in_channels, 'n_outputs': args['n_outputs'], 'bn_momentum': args['bn_momentum']}

        if is_se_net:
            params['reduction_ratio'] = args['reduction_ratio']


        data_loader = DataLoader(ds, batch_size=batch_size)
        model = ModelFactory().create_model(net, params)

        # find model dict
        for model_dir in args['model_path'].iterdir():
            
            if net == 'SEResNet':
                if (args['mask'] in str(model_dir) and \
                    ('lr_'+str(args['lr']) in str(model_dir)) and \
                    ('WD_'+str(args['weight_decay']) in str(model_dir)) and \
                    str(seed) in str(model_dir) and \
                    ('reduction'+str(args['reduction_ratio'])) in str(model_dir)):
                    model_file = model_dir / 'checkpoints' / 'Best.pt'
            elif net == 'ConcatHNN':

                end_str = args['drop_string']
#               
                if (args['mask'] in str(model_dir) and \
                    ('lr_'+str(args['lr']) in str(model_dir)) and \
                    ('WD_'+str(args['weight_decay']) in str(model_dir)) and \
                    str(seed) in str(model_dir)) and \
                    str(model_dir).endswith(end_str):
                    model_file = model_dir / 'checkpoints' / 'Best.pt'
            else:
                if (args['mask'] in str(model_dir) and \
                    ('lr_'+str(args['lr']) in str(model_dir)) and \
                    ('WD_'+str(args['weight_decay']) in str(model_dir)) and \
                    str(seed) in str(model_dir)):
                    model_file = model_dir / 'checkpoints' / 'Best.pt'

        assert (os.path.exists(model_file) and os.path.isfile(model_file)), "no model available!"

        print(f"Testing model {model_file}")

        model.load_state_dict(torch.load(model_file))
        model = model.to(device)

        predictor = Predictor(model, data_loader, device=device)

        outputs = defaultdict(list)

        for pred in predictor.predict():

            for k, v in pred.items():
                outputs[k].append(v.cpu().detach().numpy())

        pred_arrays = {}
        for k, v in outputs.items():

            last_dim_size = v[0].shape[-1]
            if last_dim_size == 1:  # check if last dimension has to be removed (exception -> multiclass logits and probabilities)
                v = [np.squeeze(vv, axis=last_dim_size) for vv in v]

            pred_arrays[k] = np.concatenate(v, axis=0)  # concat all batches

        metrics = {
               'accuracy': accuracy_score(y_true=pred_arrays["label"], y_pred=pred_arrays["classes"]),
               'balanced_accuracy': balanced_accuracy_score(y_true=pred_arrays["label"], y_pred=pred_arrays["classes"]),
               'confusion_matrix': confusion_matrix(y_true=pred_arrays["label"], y_pred=pred_arrays["classes"])}

        df.append([args['mask'], args['hippocampus'], args['lr'], seed, model_file, metrics["accuracy"], metrics["balanced_accuracy"], metrics['confusion_matrix']])
        print("Seed: {}, Mask: {}, Crop: {}, Accuracy {:4f}, Balanced_Acc {:4f}".format(seed, args['mask'], args['hippocampus'], metrics["accuracy"], metrics["balanced_accuracy"]))

        conf_matrix[0].append(pred_arrays["label"])
        conf_matrix[1].append(pred_arrays["classes"])

    df = pd.DataFrame(np.array(df), columns=['volume', 'crop', 'lr', 'seed', 'model_file', 'accuracy', 'balanced_accuracy', 'confusion_matrix'])
    df.to_csv(args['model_path'] / 'results_on_testset.csv')
    cf_mat_result = confusion_matrix(y_true=np.concatenate(conf_matrix[0]), y_pred=np.concatenate(conf_matrix[1]))
    make_confusion_matrix(cf_mat_result, sum_stats=False, file_path=(args['model_path'] / 'confusion_matrix.png'))
    
    return df