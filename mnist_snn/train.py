import brainpy_datasets as bd
import brainpy as bp
import brainpy.math as bm
from tqdm import trange


def get_MNIST_data(data_path):
    train_data = bd.vision.MNIST(data_path, split='train', download=True)
    test_data = bd.vision.MNIST(data_path, split='test', download=True)
    x_train = bm.asarray(train_data.data / 255, dtype=bm.float_).reshape(-1, 28 * 28)
    y_train = bm.asarray(train_data.targets, dtype=bm.int_)
    x_test = bm.asarray(test_data.data / 255, dtype=bm.float_).reshape(-1, 28 * 28)
    y_test = bm.asarray(test_data.targets, dtype=bm.int_)

    return x_train, y_train, x_test, y_test


class Trainer:
    def __init__(self, model, optimizer, train_config):
        self.optimizer = optimizer
        self.train_config = train_config
        self.model = model
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


    def mse_loss(self, out_fr, ys):
        return bp.losses.mean_squared_error(out_fr, ys)
    

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
        self.model.reset_state(batch_size=xs.shape[0])
        xs = self.model.encoder(xs, num_step=self.train_config.T)
        # shared arguments for looping over time
        shared = bm.shared_args_over_time(num_step=self.train_config.T)
        # outs has keys ['spike', 'inp', 'V'] under ['e_neu'] and ['i_neu'] 
        outs = bm.for_loop(self.model, (shared, xs), jit=True)
        out_fr = bm.mean(outs['e_neu']['spike'][-1], axis=0)
        ys_onehot = bm.one_hot(ys, 10, dtype=bm.float_)
        loss = self.mse_loss(out_fr, ys_onehot)

        if self.train_config.toggle_global_balance_reg_w_l2:
            reg_scale = self.train_config.global_balance_reg_scale
            global_balance_reg = reg_scale * self.global_balance_regularization_w_l2()
        else:
            global_balance_reg = bm.array([0.])

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

        n = bm.sum(out_fr.argmax(1) == ys)
        loss_info = {
            "mse_loss": loss, 
            "global_balance_reg": global_balance_reg, 
            "detailed_balance_l1_reg_sp": detailed_l1_reg_sp,
            "detailed_balance_l2_reg_sp": detailed_l2_reg_sp,
            "detailed_balance_reg_inp": detailed_balance_reg_inp,
            "n_correct": n,
        }
        total_loss = loss + global_balance_reg + detailed_balance_reg_sp + detailed_balance_reg_inp

        return total_loss, loss_info


    @bm.cls_jit
    def optimizer_step(self, xs, ys):
        grads, l, linfo = self.grad_fun(xs, ys)
        self.optimizer.update(grads)
        return l, linfo
    

    def train_epoch(self, x_train, y_train):
        print("Training...")
        bm.random.shuffle(x_train, key=123)
        bm.random.shuffle(y_train, key=123)

        loss, train_acc = [], 0.
        for i in trange(0, x_train.shape[0], self.train_config.batch):
            X = x_train[i: i + self.train_config.batch]
            Y = y_train[i: i + self.train_config.batch]
            l, linfo = self.optimizer_step(X, Y)
            loss.append(l)
            train_acc += linfo['n_correct']

        train_acc /= x_train.shape[0]
        train_loss = bm.mean(bm.asarray(loss))
        self.optimizer.lr.step_epoch()

        self.hist_train_loss.append(train_loss)
        self.hist_train_acc.append(train_acc)

        return train_loss, train_acc
    

    def validate_epoch(self, x_test, y_test):
        print("Validating...")
        loss, test_acc = [], 0.
        for i in trange(0, x_test.shape[0], self.train_config.batch):
            X = x_test[i: i + self.train_config.batch]
            Y = y_test[i: i + self.train_config.batch]
            l, linfo = self.calculate_loss(X, Y)
            loss.append(l)
            test_acc += linfo['n_correct']

        test_acc /= x_test.shape[0]
        test_loss = bm.mean(bm.asarray(loss))

        self.hist_val_loss.append(test_loss)
        self.hist_val_acc.append(test_acc)

        return test_loss, test_acc


