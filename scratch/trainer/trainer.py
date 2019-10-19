# coding: utf-8

import numpy as np


class Trainer:

    def __init__(self, model, optimizer):
        self._model = model
        self._optimizer = optimizer

    def fit(self, x, t, max_epoch=10, batch_size=32, eval_interval=10, max_grad=None):
        """

        :param x: train data
        :param t: true label
        :param max_epoch: maximum epoch for train
        :param batch_size: batch size for train
        :param eval_interval: evaluation interval step for train
        :param max_grad:
        :return:
        """
        data_size = len(x)
        max_iters = data_size // batch_size  # abandon remained data
        total_loss = 0
        loss_count = 0
        loss_list = []

        for epoch in range(max_epoch):
            rnd_idx = np.random.permutation(data_size)
            x = x[rnd_idx]
            t = t[rnd_idx]

            for step in range(max_iters):
                batch_open_index = step * batch_size
                batch_closed_index = batch_open_index + batch_size
                batch_x = x[batch_open_index:batch_closed_index]
                batch_t = t[batch_open_index:batch_closed_index]

                loss = self._model.forward(batch_x, batch_t)
                self._model.backward()
                self._optimizer.update(self._model.params, self._model.grads)

                total_loss += loss
                loss_count += 1

                if (step + 1) % 10 == 0:
                    avg_loss = total_loss / loss_count
                    print('| epoch %d | iteration step %d / %d | loss %.2f'.format(
                        epoch + 1,
                        step + 1,
                        max_iters,
                        avg_loss
                    ))
                    loss_list.append(avg_loss)
                    total_loss, loss_count = 0, 0
