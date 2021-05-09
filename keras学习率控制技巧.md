# keras学习率控制技巧

**简述思考**

在深度学习训练模型的过程中，我们通常倾向于随着训练的过程逐渐的缩小学习率来提升训练的效果。  
但是这也有一些问题，就是如果模型收敛于一个局部最小解，小的学习率可能无法从这个局部最小解  
中跳脱出来，这时候使用大的学习率，对于跳出局部最小解，更有实际价值。

**实现介绍**

根据上述的问题，我们可以在模型训练调小学习率的过程中，当学习率 < 某个特定的小值时，将学习  
率重新调整回到最初的大学习率，继续往下训练，直到训练完成，这样就可以让训练过程有更多探索的  
机会，当然过去具有较高验证精度的中间模型结果，可以保存下来，以防止探索失败

**代码**

下面的代码是对keras中ReduceLROnPlateau类的继承修改来实现我们上述的想法。

~~~python
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras import backend as K
from tensorflow.python.platform import tf_logging as logging

class LRControler(ReduceLROnPlateau):

    def __init__(self,
               monitor='val_loss',
               factor=0.1,
               patience=10,
               verbose=0,
               mode='auto',
               min_delta=1e-4,
               cooldown=0,
               min_lr=0,
               **kwargs):
        """
        :param min_lr: min_lr 作为最小学习率的控制阈值
        :param kwargs:
        """
        super(LRControler, self).__init__(monitor=monitor,
               factor=factor,
               patience=patience,
               verbose=verbose,
               mode=mode,
               min_delta=min_delta,
               cooldown=cooldown,
               min_lr=min_lr,
               **kwargs)
        return

    def on_train_begin(self, logs=None):
        super(LRControler, self).on_train_begin()
        #获取原始的学习率并保存起来，方便后续进行修复
        self.origin_lr = float(K.get_value(self.model.optimizer.lr))
        return

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        current = logs.get(self.monitor)
        if current is None:
            logging.warning('Learning rate reduction is conditioned on metric `%s` '
                            'which is not available. Available metrics are: %s',
                            self.monitor, ','.join(list(logs.keys())))

        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    old_lr = float(K.get_value(self.model.optimizer.lr))
                    if old_lr > self.min_lr:
                        new_lr = old_lr * self.factor
                        if new_lr < self.min_lr:
                            new_lr = self.origin_lr
                            print("revert lr to origin lr:", new_lr)
                        K.set_value(self.model.optimizer.lr, new_lr)
                        if self.verbose > 0:
                            print('\nEpoch %05d: ReduceLROnPlateau reducing learning '
                                  'rate to %s.' % (epoch + 1, new_lr))
                        self.cooldown_counter = self.cooldown
                        self.wait = 0
~~~
