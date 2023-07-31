import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.federated.nnUNetTrainerPartialLabelledFL import nnUNetTrainerPartialLabelledFL
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.variants.bayesian.mcmc_optimizers import SGLD, pSGLD, H_SA_SGHMC, SGHMC

def add_noise(model, lr):
    for n in [x for x in model.parameters() if len(x.size()) == 4]:
        noise = torch.randn_like(n) * 2 * lr # / np.sqrt(lr)
        #noise = noise.type(dtype)
        n.data = n.data + noise

class nnUNetTrainerMCMCBase(nnUNetTrainer):
    num_burn_in_steps = 15000
    norm_sigma = 200.
    save_interval = 100
    save_epoch_start = 70
    current_step = 0

    def run_training(self):
        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
                if batch_id % self.save_interval == 0 and epoch > self.save_epoch_start:
                    torch.save(self.network.state_dict(), join(self.output_folder, f'mcmc_ckpt_e{epoch}_b{batch_id}.pt'))

            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

        self.on_train_end()

class nnUNetTrainerPartialLabelledFLMCMCBase(nnUNetTrainerPartialLabelledFL):
    num_burn_in_steps = 20000
    norm_sigma = 200.
    save_interval = 100
    save_epoch_start = 80
    current_step = 0

    def run_training(self):
        nnUNetTrainerMCMCBase.run_training(self)

class nnUNetTrainerSGLD(nnUNetTrainerMCMCBase):
    # def configure_optimizers(self):
    #     # self.initial_lr = 1e-3
    #     optimizer = SGLD(self.network.parameters(), self.initial_lr, norm_sigma=self.norm_sigma, num_burn_in_steps=self.num_burn_in_steps)
    #     # lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
    #     lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    #     return optimizer, lr_scheduler
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
        # lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        return optimizer, lr_scheduler

    def train_step(self, batch: dict) -> dict:
        results = nnUNetTrainer.train_step(self, batch)
        if self.current_step > self.num_burn_in_steps:
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            add_noise(self.network, lr)
        self.current_step += 1
        return results

# class nnUNetTrainerpSGLD(nnUNetTrainer):
#     def configure_optimizers(self):
#         optimizer = pSGLD(self.network.parameters(), self.initial_lr, norm_sigma=0.1)
#         lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
#         return optimizer, lr_scheduler

# class nnUNetTrainerSGHMC(nnUNetTrainer):
#     def configure_optimizers(self):
#         optimizer = SGHMC(self.network.parameters(), self.initial_lr, norm_sigma=0.1)
#         lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
#         return optimizer, lr_scheduler

# class nnUNetTrainer_H_SA_SGMCMC(nnUNetTrainer):
#     # NOT WORKING YET: optmizer.step takes additional params!
#     def configure_optimizers(self):
#         optimizer = H_SA_SGHMC(self.network.parameters(), self.initial_lr)
#         lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
#         return optimizer, lr_scheduler

class nnUNetTrainerPartialLabelledFLSGLD(nnUNetTrainerPartialLabelledFLMCMCBase):
    # def configure_optimizers(self):
    #     self.initial_lr = 1e-3
    #     optimizer = SGLD(self.network.parameters(), self.initial_lr, norm_sigma=0.1)
    #     lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
    #     return optimizer, lr_scheduler
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
        # lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        return optimizer, lr_scheduler

    def train_step(self, batch: dict) -> dict:
        results = nnUNetTrainer.train_step(self, batch)
        if self.current_step > self.num_burn_in_steps:
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            add_noise(self.network, lr)
        self.current_step += 1
        return results

# class nnUNetTrainerPartialLabelledFLpSGLD(nnUNetTrainerPartialLabelledFL):
#     def configure_optimizers(self):
#         optimizer = pSGLD(self.network.parameters(), self.initial_lr, norm_sigma=0.1)
#         lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
#         return optimizer, lr_scheduler

# class nnUNetTrainerPartialLabelledFLSGHMC(nnUNetTrainerPartialLabelledFL):
#     def configure_optimizers(self):
#         optimizer = SGHMC(self.network.parameters(), self.initial_lr, norm_sigma=0.1)
#         lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
#         return optimizer, lr_scheduler

# class nnUNetTrainerPartialLabelledFL_H_SA_SGMCMC(nnUNetTrainerPartialLabelledFL):
#     # NOT WORKING YET: optmizer.step takes additional params!
#     def configure_optimizers(self):
#         optimizer = H_SA_SGHMC(self.network.parameters(), self.initial_lr)
#         lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
#         return optimizer, lr_scheduler