import detectron2.utils.comm as comm
from detectron2.engine import HookBase
import torch
from adet.utils.nnunet_generator import get_generator
from unet_pred import model_pred, pred_batch
from monai.metrics import compute_meandice

def build_val_hook(cfg):
    # same bs in nnunet for train and val loader (dk why)
    batch_num  = cfg.VAL.BATCH_NUM
    period = cfg.VAL.PERIOD
    val_loader = get_generator(cfg, 'val')

    return ValHook(period, val_loader, batch_num)

def dc(res, gt):
    '''
    res, gt: 5d data'''
    res, gt = res.bool(), gt.bool()
    all0 = (res.sum(dim=[-1,-2,-3])==0)
    all0gt = (gt.sum(dim=[-1,-2,-3])==0)
    dce = compute_meandice(res, gt, False)
    if all0gt.any():
        dce[all0 & all0gt] = 1
        dce[all0 ^ all0gt] = 0
    if dce.isnan().any():
        print(dce, all0, all0gt)
    return dce

class ValHook(HookBase):
    """
    Run an evaluation function periodically, and at the end of training.

    It is executed every ``eval_period`` iterations and after the last iteration.
    """

    def __init__(self, eval_period, loader, batch_num):
        """
        Args:
            eval_period (int): the period to run `eval_function`. Set to 0 to
                not evaluate periodically (but still after the last iteration).
            eval_function (callable): a function which takes no arguments, and
                returns a nested dict of evaluation metrics.

        Note:
            This hook must be enabled in all or none workers.
            If you would like only certain workers to perform evaluation,
            give other workers a no-op function (`eval_function=lambda: None`).
        """
        self._period = eval_period
        self._loader = loader
        self._iter_num = batch_num

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if self._period > 0 and next_iter % self._period == 0:
            # do the last eval in after_train
            if next_iter != self.trainer.max_iter:
                self._do_eval()
    
    def _do_eval(self):
        model = self.trainer.model
        is_train = model.training
        val_dc = []
        for i in range(self._iter_num):
            data = next(self._loader)
            img, gt = data['data'], data['target']
            # print(img.shape, gt.shape)
            with torch.no_grad():
                model.eval()
                res = model_pred(img, model)
                res = res[:, 1:] # 0 is for bkgrd
                # gt = (gt==1)
                gt = gt>0
                dce = dc(res, gt.cuda())
                # print(dce.shape)
                val_dc.append(dce)

        val_dc = torch.cat(val_dc).mean(dim=0)
        loss_dict = {'est_dc': val_dc}
        losses = sum(loss_dict.values())
        assert torch.isfinite(losses).all(), loss_dict

        loss_dict_reduced = {"val_" + k: v.item() for k, v in 
                            comm.reduce_dict(loss_dict).items()}
        if comm.is_main_process():
            self.trainer.storage.put_scalars(**loss_dict_reduced)
            print(loss_dict_reduced)
            print()
        model.train(is_train)

    def after_train(self):
        # This condition is to prevent the eval from running after a failed training
        if self.trainer.iter + 1 >= self.trainer.max_iter:
            self._do_eval()
        # func is likely a closure that holds reference to the trainer
        # therefore we clean it to avoid circular reference in the end
        del self._loader