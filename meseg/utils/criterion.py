import torch
from torch import nn
from torch.nn import functional as F, BCEWithLogitsLoss, MSELoss
from torch.cuda.amp import GradScaler
from monai.losses import DiceCELoss as monaiDiceCELoss



class NativeScalerWithGradAccum:
    def __init__(self):
        """NativeScalerWithGradAccum (timm)
        Native(pytorch) f16 scaler
        """
        self._scaler = GradScaler()

    def __call__(self, loss, optimizer, model_param, scheduler=None, grad_norm=None, update=True):
        self._scaler.scale(loss).backward()
        if update:
            if grad_norm:
                self._scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model_param, grad_norm)
            self._scaler.step(optimizer)
            self._scaler.update()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)



class BCEWithLogitsLossWithTypeCasting(BCEWithLogitsLoss):
    def forward(self, y_hat, y):
        y = y.float()
        y = y.reshape(y_hat.shape)
        return super().forward(y_hat, y)



class DiceLoss(nn.Module):
    def __init__(self, n_classes:int, softmax:bool):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.softmax = softmax

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None):
        if self.softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes



class DiceCELoss(nn.Module):
    def __init__(self, num_classes:int, p:float, softmax:bool) -> None:
        super(DiceCELoss, self).__init__()
        self.dice_loss = DiceLoss(num_classes, softmax)
        self.ce_loss = nn.CrossEntropyLoss()
        self.p = p
    
    def forward(self, y_hat:torch.Tensor, y:torch.Tensor)->torch.Tensor:
        """
        y_hat is prediected tensor.
        y is ground truth tensor.
        """
        total_loss = self.ce_loss(y_hat, y)*self.p + self.dice_loss(y_hat, y)*(1-self.p)
        return total_loss



def get_criterion_scaler(args):
    """Get Criterion(Loss) function and scaler
    Criterion functions are divided depending on usage of mixup
    - w/ mixup - you don't need to add smoothing loss, because mixup will add smoothing loss.
    - w/o mixup - you should need to add smoothing loss
    """

    if args.criterion in ['ce', 'crossentropy']:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.smoothing)
    elif args.criterion in ['bce', 'binarycrossentropy']:
        criterion = BCEWithLogitsLossWithTypeCasting()
    elif args.criterion in ['mse', 'l2']:
        criterion = MSELoss()
    elif args.criterion in ['dicece', 'diceceloss']:
        if args.model_name in ["UNETR"]:
            criterion = monaiDiceCELoss(to_onehot_y=True, softmax=True)
        else:
            criterion = DiceCELoss(args.num_classes, 0.5, softmax=True)

    if args.amp:
        scaler = NativeScalerWithGradAccum()
    else:
        scaler = None

    return criterion, scaler