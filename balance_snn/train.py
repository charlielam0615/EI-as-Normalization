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


def mse_loss(out_fr, ys):
    return bp.losses.mean_squared_error(out_fr, ys)


def global_balance_regularization(model, kappa):
    reg = bm.Variable(0.)
    for i in range(model.n_layer):
        for wp in model.w_pattern:
            w = getattr(model, wp.format(i)).W
            w_l2 = bm.sqrt(bm.sum(bm.square(w)))
            reg += bm.square(w_l2 - kappa)
    return reg


def calculate_loss(xs, ys, model, loss_config, encoder):
    model.reset_state(batch_size=xs.shape[0])
    xs = encoder(xs, num_step=loss_config.T)
    # shared arguments for looping over time
    shared = bm.shared_args_over_time(num_step=loss_config.T)
    outs = bm.for_loop(model, (shared, xs))
    out_fr = bm.mean(outs, axis=0)
    ys_onehot = bm.one_hot(ys, 10, dtype=bm.float_)
    loss = mse_loss(out_fr, ys_onehot)

    if loss_config.toggle_global_balance_reg:
        reg_scale = loss_config.global_balance_reg_scale
        global_balance_reg = reg_scale * global_balance_regularization(model, loss_config.kappa)
    else:
        global_balance_reg = 0.

    n = bm.sum(out_fr.argmax(1) == ys)
    return loss + global_balance_reg, n


# train
@bm.jit(static_argnums=(2, 3))
def optimizer_step(xs, ys, grad_fun, optimizer):
    grads, l, n = grad_fun(xs, ys)
    optimizer.update(grads)
    return l, n
    

def train_epoch(x_train, y_train, grad_fun, optimizer, train_config):
    print("Training...")
    bm.random.shuffle(x_train, key=123)
    bm.random.shuffle(y_train, key=123)

    loss, train_acc = [], 0.
    for i in trange(0, x_train.shape[0], train_config.batch):
        X = x_train[i: i + train_config.batch]
        Y = y_train[i: i + train_config.batch]
        l, correct_num = optimizer_step(X, Y, grad_fun, optimizer)
        loss.append(l)
        train_acc += correct_num

    train_acc /= x_train.shape[0]
    train_loss = bm.mean(bm.asarray(loss))
    optimizer.lr.step_epoch()

    return train_loss, train_acc


def validate_epoch(x_test, y_test, loss_fun, global_config):
    print("Validating...")
    loss, test_acc = [], 0.
    for i in trange(0, x_test.shape[0], global_config.batch):
        X = x_test[i: i + global_config.batch]
        Y = y_test[i: i + global_config.batch]
        l, correct_num = loss_fun(X, Y)
        loss.append(l)
        test_acc += correct_num

    test_acc /= x_test.shape[0]
    test_loss = bm.mean(bm.asarray(loss))

    return test_loss, test_acc


