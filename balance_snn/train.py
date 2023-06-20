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
        self.grad_fun = bm.grad(
            self.calculate_loss, 
            grad_vars=model.train_vars().unique(), 
            has_aux=True, 
            return_value=True
        )

    def mse_loss(self, out_fr, ys):
        return bp.losses.mean_squared_error(out_fr, ys)
    
    def global_balance_regularization(self):
        reg = bm.Variable(0.)
        for i in range(self.model.n_layer):
            for wp in self.model.w_pattern:
                w = getattr(self.model, wp.format(i)).W
                w_l2 = bm.sqrt(bm.sum(bm.square(w)))
                reg += bm.square(w_l2 - self.train_config.kappa)
        return reg

    def calculate_loss(self, xs, ys):
        self.model.reset_state(batch_size=xs.shape[0])
        xs = self.model.encoder(xs, num_step=self.train_config.T)
        # shared arguments for looping over time
        shared = bm.shared_args_over_time(num_step=self.train_config.T)
        outs = bm.for_loop(self.model, (shared, xs))
        out_fr = bm.mean(outs, axis=0)
        ys_onehot = bm.one_hot(ys, 10, dtype=bm.float_)
        loss = self.mse_loss(out_fr, ys_onehot)

        if self.train_config.toggle_global_balance_reg:
            reg_scale = self.train_config.global_balance_reg_scale
            global_balance_reg = reg_scale * self.global_balance_regularization()
        else:
            global_balance_reg = 0.

        n = bm.sum(out_fr.argmax(1) == ys)
        return loss + global_balance_reg, n

    @bm.cls_jit
    def optimizer_step(self, xs, ys):
        grads, l, n = self.grad_fun(xs, ys)
        self.optimizer.update(grads)
        return l, n
    

    def train_epoch(self, x_train, y_train):
        print("Training...")
        bm.random.shuffle(x_train, key=123)
        bm.random.shuffle(y_train, key=123)

        loss, train_acc = [], 0.
        for i in trange(0, x_train.shape[0], self.train_config.batch):
            X = x_train[i: i + self.train_config.batch]
            Y = y_train[i: i + self.train_config.batch]
            l, correct_num = self.optimizer_step(X, Y)
            loss.append(l)
            train_acc += correct_num

        train_acc /= x_train.shape[0]
        train_loss = bm.mean(bm.asarray(loss))
        self.optimizer.lr.step_epoch()

        return train_loss, train_acc

    def validate_epoch(self, x_test, y_test):
        print("Validating...")
        loss, test_acc = [], 0.
        for i in trange(0, x_test.shape[0], self.train_config.batch):
            X = x_test[i: i + self.train_config.batch]
            Y = y_test[i: i + self.train_config.batch]
            l, correct_num = self.calculate_loss(X, Y)
            loss.append(l)
            test_acc += correct_num

        test_acc /= x_test.shape[0]
        test_loss = bm.mean(bm.asarray(loss))

        return test_loss, test_acc


