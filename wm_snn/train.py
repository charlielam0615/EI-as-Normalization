import brainpy as bp
import brainpy.math as bm
import brainpy_datasets as bd
import numpy as np
from tqdm import tqdm

from wm_snn.wm import DMS

def get_WM_data():
    ds = DMS(dt=bm.dt, mode='spiking', num_trial=64 * 50, bg_fr=1.)
    train_loader = bd.cognitive.TaskLoader(ds, shuffle=True, batch_size=64, data_first_axis='T')
    ds = DMS(dt=bm.dt, mode='spiking', num_trial=64 * 20, bg_fr=1.)
    test_loader = bd.cognitive.TaskLoader(ds, shuffle=True, batch_size=64, data_first_axis='T')
    return train_loader, test_loader


class Trainer:
    def __init__(self, model, optimizer, train_config, t_test):
        self.optimizer = optimizer
        self.train_config = train_config
        self.model = model
        self.t_test = t_test
        self.hist_train_loss = []
        self.hist_train_acc = []
        self.hist_val_loss = []
        self.hist_val_acc = []
        self.grad_fun = bm.grad(
            self.calculate_loss, 
            grad_vars=model.train_vars().unique(), 
            has_aux=True, 
            return_value=True
        )


    def ce_loss(self, out_fr, y_label):
        tiled_targets = bm.tile(bm.expand_dims(y_label, 0), (self.t_test, 1))
        # TODO: shoud transpose?
        return bp.losses.cross_entropy_loss(out_fr, tiled_targets, reduction='mean')
    
    
    def mse_loss(self, out_fr, y_onehot):
        return bp.losses.mean_squared_error(out_fr, y_onehot)
    

    def global_balance_regularization_w_l2(self):
        reg = bm.Variable(0.)
        for i in range(self.model.n_layer):
            for wp in self.model.w_pattern:
                w = getattr(self.model, wp.format(i)).W
                w_l2 = bm.sqrt(bm.sum(bm.square(w)))
                reg += bm.square(w_l2 - self.train_config.kappa)
        return reg
    

    def detailed_balance_regularization_l1_l2_spike(self, outs):
        l1_reg = bm.Variable(0.)
        l2_reg = bm.Variable(0.)
        for neu in ['e_neu', 'i_neu']:
            for neu_sp in outs[neu]['spike']:
                l1_reg += bm.mean(bm.sum(bm.abs(neu_sp-0.01), axis=[0, 2], keepdims=True))
                l2_reg += bm.mean(bm.sum(bm.square(bm.sum(neu_sp, axis=0, keepdims=True)), axis=2))
        return l1_reg, l2_reg
    
    
    def detailed_balance_regularization_input(self, outs):
        reg = bm.Variable(0.)
        for neu in ['e_neu', 'i_neu']:
            for inp in outs[neu]['inp']:
                reg += bm.mean(bm.sum(bm.abs(inp), axis=0))
        return reg


    @bm.cls_jit
    def calculate_loss(self, xs, ys):
        self.model.reset_state(batch_size=xs.shape[1])
        # shared arguments for looping over time
        shared = bm.shared_args_over_time(num_step=self.train_config.T)
        # outs has keys ['spike', 'inp'] under ['e_neu'] and ['i_neu'] 
        outs = bm.for_loop(self.model, (shared, xs), jit=True)
        out_fr = outs['e_neu']['spike'][-1][-self.t_test:]
        # ys_onehot = bm.one_hot(ys, 2, dtype=bm.float_)
        loss = self.ce_loss(out_fr, ys)
        if self.train_config.toggle_global_balance_reg_w_l2:
            reg_scale = self.train_config.global_balance_reg_scale
            global_balance_reg = reg_scale * self.global_balance_regularization_w_l2()
        else:
            global_balance_reg = 0.

        if self.train_config.toggle_detailed_balance_reg_l1_l2_spike:
            l1_reg_scale = self.train_config.detailed_balance_l1_reg_sp_scale
            l2_reg_scale = self.train_config.detailed_balance_l2_reg_sp_scale
            detailed_l1_reg_sp, detailed_l2_reg_sp = self.detailed_balance_regularization_l1_l2_spike(outs)
            detailed_l1_reg_sp, detailed_l2_reg_sp = \
                l1_reg_scale * detailed_l1_reg_sp, l2_reg_scale * detailed_l2_reg_sp
            detailed_balance_reg_sp = detailed_l1_reg_sp + detailed_l2_reg_sp
        else:
            detailed_balance_reg_sp, detailed_l1_reg_sp, detailed_l2_reg_sp = 0., 0., 0.
            
        if self.train_config.toggle_detailed_balance_reg_input:
            detailed_l1_reg_inp = self.detailed_balance_regularization_input(outs)
            l1_reg_scale = self.train_config.detailed_balance_l1_reg_inp_scale
            detailed_l1_reg_inp = l1_reg_scale * detailed_l1_reg_inp
            detailed_balance_reg_inp = detailed_l1_reg_inp
        else:
            detailed_balance_reg_inp, detailed_l1_reg_inp = 0., 0.

        loss_reg_v = 0.
        if self.train_config.toggle_membrane_reg:
            for i in range(self.model.n_layer):
                for neu in ['e_neu', 'i_neu']:
                    loss_reg_v += self.train_config.membrane_reg_scale * \
                        bm.square(bm.mean(
                            bm.relu(outs[neu]['V'][i] - 0.4) ** 2 +
                            bm.relu(-(outs[neu]['V'][i] - (-2))) ** 2
                        ))

        n = bm.sum(bm.equal(ys, bm.argmax(bm.mean(out_fr, axis=0), axis=1)))

        loss_info = {
            "ce_loss": loss, 
            "global_balance_reg": global_balance_reg, 
            "detailed_balance_l1_reg_sp": detailed_l1_reg_sp,
            "detailed_balance_l2_reg_sp": detailed_l2_reg_sp,
            "detailed_balance_reg_inp": detailed_balance_reg_inp,
            "loss_reg_v": loss_reg_v,
            "n_correct": n,
        }
        total_loss = loss + global_balance_reg + detailed_balance_reg_sp + detailed_balance_reg_inp + loss_reg_v

        return total_loss, loss_info


    @bm.cls_jit
    def optimizer_step(self, xs, ys):
        grads, l, linfo = self.grad_fun(xs, ys)
        self.optimizer.update(grads)
        return l, linfo
    

    def train_epoch(self, train_loader):
        print("Training...")

        loss, train_acc = [], 0.
        for X, Y in tqdm(train_loader):
            l, linfo = self.optimizer_step(X, Y)
            loss.append(l)
            train_acc += linfo['n_correct']

        train_acc /= len(train_loader.dataset)
        train_loss = bm.mean(bm.asarray(loss))
        self.optimizer.lr.step_epoch()

        self.hist_train_loss.append(train_loss)
        self.hist_train_acc.append(train_acc)

        return train_loss, train_acc
    

    def validate_epoch(self, test_loader):
        print("Validating...")
        loss, test_acc = [], 0.
        for X, Y in tqdm(test_loader):
            l, linfo = self.calculate_loss(X, Y)
            loss.append(l)
            test_acc += linfo['n_correct']

        test_acc /= len(test_loader.dataset)
        test_loss = bm.mean(bm.asarray(loss))

        self.hist_val_loss.append(test_loss)
        self.hist_val_acc.append(test_acc)

        return test_loss, test_acc


