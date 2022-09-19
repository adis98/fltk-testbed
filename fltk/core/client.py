from typing import Tuple, Any

import numpy as np
import sklearn
import time
import torch

from fltk.core.node import Node
from fltk.schedulers import MinCapableStepLR
from fltk.strategy import get_optimizer
from fltk.util.config import FedLearningConfig
from fltk.samplers.unique import mapper
from fltk.nets import Nets


class Client(Node):
    """
    Federated experiment client.
    """
    running = False

    def __init__(self, identifier: str, rank: int, world_size: int, config: FedLearningConfig):
        super().__init__(identifier, rank, world_size, config)

        self.loss_function = self.config.get_loss_function()()
        self.optimizer = get_optimizer(self.config.optimizer)(self.net.parameters(),
                                                              **self.config.optimizer_args)
        self.scheduler = MinCapableStepLR(self.optimizer,
                                          self.config.scheduler_step_size,
                                          self.config.scheduler_gamma,
                                          self.config.min_lr)

        # init variables for continual learning
        self.continual = config.continual
        self.task_cnt = config.task_cnt
        self.rounds_per_task = config.rounds_per_task
        self.window = config.output_window_type
        self.classes_per_task = config.classes_per_task
        self.model_name = config.net_name
        self.lambda_l2 = config.lambda_l2
        self.lambda_l1 = config.lambda_l1
        self.wd = config.weight_decay
        self.lambda_mask = config.lambda_mask
        self.pre_weight = None
        if self.model_name == Nets.cifar100_lenetweit:
            self.pre_weight = {
                'weight': [],
                'aw': [],
                'mask': []
            }

    def remote_registration(self):
        """
        Function to perform registration to the remote. Currently, this will connect to the Federator Client. Future
        version can provide functionality to register to an arbitrary Node, including other Clients.
        @return: None.
        @rtype: None
        """
        self.logger.info('Sending registration')
        self.message('federator', 'ping', 'new_sender')
        self.message('federator', 'register_client', self.id, self.rank)
        self.running = True
        self._event_loop()

    def stop_client(self):
        """
        Function to stop client after training. This allows remote clients to stop the client within a specific
        timeframe.
        @return: None
        @rtype: None
        """
        self.logger.info('Got call to stop event loop')
        self.running = False

    def _event_loop(self):
        self.logger.info('Starting event loop')
        while self.running:
            time.sleep(0.1)
        self.logger.info('Exiting node')

    def train(self, num_epochs: int, round_id: int):
        """
        Function implementing federated learning training loop, allowing to run for a configurable number of epochs
        on a local dataset. Note that only the last statistics of a run are sent to the caller (i.e. Federator).
        @param num_epochs: Number of epochs to run during a communication round's training loop.
        @type num_epochs: int
        @param round_id: Global communication round ID to be used during training.
        @type round_id: int
        @return: Final running loss statistic and acquired parameters of the locally trained network. NOTE that
        intermediate information is only logged to the STD-out.
        @rtype: Tuple[float, Dict[str, torch.Tensor]]
        """
        start_time = time.time()
        running_loss = 0.0
        final_running_loss = 0.0
        for local_epoch in range(num_epochs):
            effective_epoch = round_id * num_epochs + local_epoch
            progress = f'[RD-{round_id}][LE-{local_epoch}][EE-{effective_epoch}]'

            # Additional fixes in case of continual learning
            task_id = None
            if self.continual:
                task_id = round_id // self.rounds_per_task
                self.dataset.train_sampler.set_task(task_id)

            if self.distributed:
                # In case a client occurs within (num_epochs) communication rounds as this would cause
                # an order or data to re-occur during training.
                self.dataset.train_sampler.set_epoch(effective_epoch)

            training_cardinality = len(self.dataset.get_train_loader())
            self.logger.info(f'{progress}{self.id}: Number of training samples: {training_cardinality}')

            for i, (inputs, labels) in enumerate(self.dataset.get_train_loader(), 0):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                if task_id is not None:
                    if self.window == "sliding":
                        outputs = self.net(inputs, task_id)
                        labels = list(map(lambda x: mapper[x][1] + task_id * self.classes_per_task, labels.numpy().astype(int)))
                        labels = torch.tensor(labels)
                    elif self.window == "expanding":
                        outputs = self.net(inputs, task_id)
                        labels = list(map(lambda x: mapper[x][1] + task_id * self.classes_per_task, labels.numpy().astype(int)))
                        labels = torch.tensor(labels)
                    else:
                        outputs = self.net(inputs, task_id)

                else:
                    outputs = self.net(inputs)

                if self.model_name == Nets.cifar100_lenetweit:
                    loss = self.get_loss(outputs, labels, task_id)
                else:
                    loss = self.loss_function(outputs, labels)

                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                # Mark logging update step
                if i % self.config.log_interval == 0:
                    self.logger.info(
                        f'[{self.id}] [{local_epoch}/{num_epochs:d}, {i:5d}] loss: {running_loss / self.config.log_interval:.3f}')
                    final_running_loss = running_loss / self.config.log_interval
                    running_loss = 0.0
            end_time = time.time()
            duration = end_time - start_time
            self.logger.info(f'{progress} Train duration is {duration} seconds')

        return final_running_loss, self.get_nn_parameters(),

    def set_tau_eff(self, total):
        client_weight = self.get_client_datasize() / total
        n = self.get_client_datasize()  # pylint: disable=invalid-name
        E = self.config.epochs  # pylint: disable=invalid-name
        B = 16  # nicely hardcoded :) # pylint: disable=invalid-name
        tau_eff = int(E * n / B) * client_weight
        if hasattr(self.optimizer, 'set_tau_eff'):
            self.optimizer.set_tau_eff(tau_eff)

    def test(self) -> Tuple[float, float, np.array]:
        """
        Function implementing federated learning test loop.
        @return: Statistics on test-set given a (partially) trained model; accuracy, loss, and confusion matrix.
        @rtype: Tuple[float, float, np.array]
        """
        start_time = time.time()
        correct = 0
        total = 0
        targets_ = []
        pred_ = []
        loss = 0.0
        with torch.no_grad():
            for (images, labels) in self.dataset.get_test_loader():
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.net(images)

                _, predicted = torch.max(outputs.data, 1)  # pylint: disable=no-member
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                targets_.extend(labels.cpu().view_as(predicted).numpy())
                pred_.extend(predicted.cpu().numpy())

                loss += self.loss_function(outputs, labels).item()
        # Calculate learning statistics
        loss /= len(self.dataset.get_test_loader().dataset)
        accuracy = 100.0 * correct / total

        confusion_mat = sklearn.metrics.confusion_matrix(targets_, pred_)

        end_time = time.time()
        duration = end_time - start_time
        self.logger.info(f'Test duration is {duration} seconds')
        return accuracy, loss, confusion_mat

    def continualtest(self, until_task) -> Tuple[float, float, np.array]:
        """
        Function implementing federated learning test loop for continual learning.
        @return: Statistics on test-set given a (partially) trained model; accuracy, loss, and confusion matrix.
        @rtype: Tuple[float, float, np.array]
        """
        start_time = time.time()
        correct = 0
        total = 0
        targets_ = []
        pred_ = []
        loss = 0.0
        with torch.no_grad():
            for task_id in range(until_task + 1):
                self.dataset.test_sampler.set_task(task_id)
                for (images, labels) in self.dataset.get_test_loader():
                    images, labels = images.to(self.device), labels.to(self.device)
                    if self.window == "sliding":
                        outputs = self.net(images, task_id)
                        labels = list(map(lambda x: mapper[x][1] + task_id * self.classes_per_task, labels.numpy().astype(int)))
                        labels = torch.tensor(labels)
                    elif self.window == "expanding":
                        outputs = self.net(images, task_id)
                        labels = list(map(lambda x: mapper[x][1] + task_id * self.classes_per_task, labels.numpy().astype(int)))
                        labels = torch.tensor(labels)
                    else:
                        outputs = self.net(images, task_id)

                    _, predicted = torch.max(outputs.data, 1)  # pylint: disable=no-member
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    targets_.extend(labels.cpu().view_as(predicted).numpy())
                    pred_.extend(predicted.cpu().numpy())

                    if self.model_name == Nets.cifar100_lenetweit:
                        loss = self.get_loss(outputs, labels, task_id)
                    else:
                        loss += self.loss_function(outputs, labels).item()
        # Calculate learning statistics
        loss /= len(self.dataset.get_test_loader().dataset)
        accuracy = 100.0 * correct / total

        confusion_mat = sklearn.metrics.confusion_matrix(targets_, pred_)

        end_time = time.time()
        duration = end_time - start_time
        self.logger.info(f'Test duration is {duration} seconds')
        return accuracy, loss, confusion_mat

    def get_client_datasize(self):  # pylint: disable=missing-function-docstring
        return len(self.dataset.get_train_sampler())

    def exec_round(self, num_epochs: int, round_id: int) -> Tuple[Any, Any, Any, Any, float, float, float, np.array]:
        """
        Function as access point for the Federator Node to kick off a remote learning round on a client.
        @param num_epochs: Number of epochs to run
        @type num_epochs: int
        @return: Tuple containing the statistics of the training round; loss, weights, accuracy, test_loss, make-span,
        training make-span, testing make-span, and confusion matrix.
        @rtype: Tuple[Any, Any, Any, Any, float, float, float, np.array]
        """
        self.logger.info(f"[EXEC] running {num_epochs} locally...")
        start = time.time()
        loss, weights = self.train(num_epochs, round_id)
        time_mark_between = time.time()

        kb = []
        mask = []
        if self.continual and ('weit' in str(self.config.net_name)):
            for name, para in self.net.named_parameters():
                if 'aw' in name:
                    aw = para.detach()
                    aw.requires_grad = False
                    kb.append(aw)
                elif 'mask' in name:
                    msk = para.detach()
                    msk.requires_grad = False
                    mask.append(msk)

            num_tasks_learnt = (round_id // self.config.rounds_per_task) + 1
            if len(self.pre_weight['aw']) < num_tasks_learnt:
                self.pre_weight['aw'].append([])
                self.pre_weight['mask'].append([])
                self.pre_weight['weight'].append([])
            self.pre_weight['aw'][-1] = kb
            self.pre_weight['mask'][-1] = mask
            self.pre_weight['weight'][-1] = self.net.get_weights()

        if self.continual:
            until_task = round_id // self.rounds_per_task  # testing to be done from task_ID zero till the until_task (inclusive)
            accuracy, test_loss, test_conf_matrix = self.continualtest(until_task)
        else:
            accuracy, test_loss, test_conf_matrix = self.test()

        end = time.time()
        round_duration = end - start
        train_duration = time_mark_between - start
        test_duration = end - time_mark_between
        # self.logger.info(f'Round duration is {duration} seconds')

        if hasattr(self.optimizer, 'pre_communicate'):  # aka fednova or fedprox
            self.optimizer.pre_communicate()
        for k, value in weights.items():
            weights[k] = value.cpu()

        if (round_id%self.config.rounds_per_task == self.config.rounds_per_task-1) and len(kb) > 0:  # return knowledge base if its the last round of task and if it actually has something, i.e., WEIT model
            return loss, weights, accuracy, test_loss, round_duration, train_duration, test_duration, test_conf_matrix, kb
        else:
            return loss, weights, accuracy, test_loss, round_duration, train_duration, test_duration, test_conf_matrix


    def __del__(self):
        self.logger.info(f'Client {self.id} is stopping')

    def l2_loss(self, para):
        return torch.sum(torch.pow(para, 2)) / 2

    def get_loss(self, outputs, labels, t):
        loss = self.loss_function(outputs, labels).item()
        i = 0
        weight_decay = 0
        sparseness = 0
        approx_loss = 0
        sw = None
        aw = None
        mask = None
        for name, para in self.net.named_parameters():
            if 'sw' in name:
                sw = para
            elif 'aw' in name:
                aw = para
            elif 'mask' in name:
                mask = para
            elif 'atten' in name:
                weight_decay += self.wd * self.l2_loss(aw)
                weight_decay += self.wd * self.l2_loss(mask)
                sparseness += self.lambda_l1 * torch.sum(torch.abs(aw))
                sparseness += self.lambda_mask * torch.sum(torch.abs(mask))
                if torch.isnan(weight_decay).sum() > 0:
                    print('weight_decay nan')
                if torch.isnan(sparseness).sum() > 0:
                    print('sparseness nan')
                if t == 0:
                    weight_decay += self.wd * self.l2_loss(sw)
                else:
                    for tid in range(t):
                        prev_aw = self.pre_weight['aw'][tid][i]
                        prev_mask = self.pre_weight['mask'][tid][i]
                        m = torch.nn.Sigmoid()
                        g_prev_mask = m(prev_mask)
                        #################################################
                        sw2 = sw.transpose(0, -1)
                        restored = (sw2 * g_prev_mask).transpose(0, -1) + prev_aw
                        a_l2 = self.l2_loss(restored - self.pre_weight['weight'][tid][i])
                        approx_loss += self.lambda_l2 * a_l2
                        #################################################
                    i += 1
        loss += weight_decay + sparseness + approx_loss
        return loss
