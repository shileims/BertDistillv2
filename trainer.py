import copy
import time
import torch
import os.path
import numpy as np
from tqdm import tqdm
from pathlib import Path
from utils import Statistics, metric_monitor, save_checkpoint, distill_metric_monitor
from torch.backends import cudnn
cudnn.benchmark = True

from utils import logger
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError as e:
    logger.log('******SummaryWrite is None!******')
    SummaryWriter = None

class Trainer(object):
    def __init__(self, args, model, optimizer, scheduler, val_dataloader, train_dataloader, criterion, loss_scaler=None):
        super(Trainer, self).__init__()
        self.args = args
        self.model = model
        self.val_dataloader = val_dataloader
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = args.device
        self.criterion = criterion
        self.loss_scaler = loss_scaler
        self.train_iterations = args.start_iteration
        logger.log(f'Training from {args.start_iteration}th iteration')
        self.v2lacc_func = lambda x, y: list({ 'V2LAcc_R' + str(yy): getattr(x, 'avg_statistics')(metric_name='V2LAcc_R' + str(yy)) } for yy in y)
        self.l2vacc_func = lambda x, y: list({ 'L2VAcc_R' + str(yy): getattr(x, 'avg_statistics')(metric_name='L2VAcc_R' + str(yy)) } for yy in y)

        self.train_metric_names = [args.train_metric_names] if isinstance(args.train_metric_names, str) else args.train_metric_names
        if 'Loss' not in self.train_metric_names:
            self.train_metric_names.append('Loss')

        self.val_metric_names = [args.val_metric_names] if isinstance(args.val_metric_names, str) else args.val_metric_names
        if 'Loss' not in self.val_metric_names:
            self.val_metric_names.append('Loss')

        self.tb_log_writter = None
        if SummaryWriter is not None:
            self.setup_log_writer()
            
        if args.amp:
            assert self.loss_scaler is not None

    def setup_log_writer(self):
        log_dir = os.path.join(self.args.output_dir, 'tb_log')
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        logger.log('Creat tb log dir!')
        self.tb_log_writter = SummaryWriter(log_dir=log_dir, comment='Training and Validation Logs')

    def run(self, train_sampler):
        if train_sampler is None:
            logger.error('Train sampler cannot be None')
        train_start_time = time.time()
        logger.log(f'Starting training from {self.args.start_epoch}th epotch')
        for epoch in range(self.args.start_epoch, self.args.epochs):
            train_sampler.set_epoch(epoch)
            # update image scales for multi-scale training
            # train_sampler.update_scales(epoch=epoch)
            train_v2laccs, train_l2vaccs, train_loss = self.train_epoch(epoch)
            val_v2laccs, val_l2vaccs, val_loss = self.val_epoch(epoch)

            if epoch % self.args.save_ckpt_freq == 0:
                save_ckpt_path = save_checkpoint(
                    train_iterations=self.train_iterations,
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    args=self.args,
                )
                logger.info('Checkpoints saved at: {}'.format(save_ckpt_path), print_line=True)

            if self.tb_log_writter is not None:
                lr_list = self.scheduler.retrieve_lr(self.optimizer)
                for g_id, lr_val in enumerate(lr_list):
                    self.tb_log_writter.add_scalar('LR/Group-{}'.format(g_id), round(lr_val, 6), epoch)
                self.tb_log_writter.add_scalar('Train/Loss', round(train_loss, 2), epoch)
                self.tb_log_writter.add_scalar('Val/Loss', round(val_loss, 2), epoch)
                for k, v in train_v2laccs.items():
                    self.tb_log_writter.add_scalar(f'Train/{k}', round(v, 2), epoch)
                for k, v in train_l2vaccs.items():
                    self.tb_log_writter.add_scalar(f'Train/{k}', round(v, 2), epoch)
                for k, v in val_v2laccs.items():
                    self.tb_log_writter.add_scalar(f'Val/{k}', round(v, 2), epoch)
                for k, v in val_l2vaccs.items():
                    self.tb_log_writter.add_scalar(f'Val/{k}', round(v, 2), epoch)

        torch.cuda.empty_cache()

        if self.tb_log_writter is not None:
            self.tb_log_writter.close()

        train_end_time = time.time()
        hours, rem = divmod(train_end_time - train_start_time, 3600)
        minutes, seconds = divmod(rem, 60)
        train_time_str = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
        logger.log('Training took {}'.format(train_time_str))

    def train_epoch(self, epoch):
        # Only achieving results for rank 1 --> rank, metric_names
        train_stats = Statistics(metric_names=self.train_metric_names, SUPPORTED_STATS=self.train_metric_names)
        if hasattr(self.model, 'module'):
            self.model.module.train()
        else:
            self.model.train()

        epoch_start_time = time.time()
        batch_load_start = time.time()
        for batch_id, batch in enumerate(tqdm(self.train_dataloader)):
            batch_load_toc = time.time() - batch_load_start
            input_img, input_text = batch[0], batch[1]
            input_img = input_img.to(self.args.device)
            input_text = {k: v.to(self.args.device, non_blocking=True) for k, v in input_text.items()}

            self.optimizer = self.scheduler.update_lr(self.optimizer, len(self.train_dataloader), epoch, batch_id)

            logits_image, logits_text = self.model(input_img, input_text)
            loss = self.criterion(logits_image, logits_text)

            # rank serves to picking up top rank similarity accuracy
            rank = self.args.train_rank
            batch_size = logits_text.size(0)
            labels = np.arange(batch_size)
            t2i_ranks = self.compute_rank_onehoc(logits_text.detach(), labels, keep_top_k=max(rank))
            i2t_ranks = self.compute_rank_onehoc(logits_image.detach(), labels, keep_top_k=max(rank))
            i2t_accs = [sum([_ < r for _ in i2t_ranks]) / len(i2t_ranks) * 100. for r in rank]
            t2i_accs = [sum([_ < r for _ in t2i_ranks]) / len(t2i_ranks) * 100. for r in rank]

            if isinstance(loss, torch.Tensor) and torch.isnan(loss):
                import pdb
                pdb.set_trace()

            before_backward_params = None
            params_name = None
            for idx, (name, params) in enumerate(self.model.named_parameters()):
                if params.requires_grad:
                    before_backward_params = copy.deepcopy(params.data)
                    params_name = name
                    break

            loss.backward()
            if (batch_id + 1) % self.args.accumulate_step == 0:
                if self.args.clip_grad is not None:
                    # For gradient clipping, unscale the gradients and then clip them
                    # self.gradient_scalar.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.clip_grad)

                self.optimizer.step()
                self.optimizer.zero_grad()

                after_backward_params = None
                for idx, (name, params) in enumerate(self.model.named_parameters()):
                    if params.requires_grad and name == params_name:
                        after_backward_params = params
                        break

                # check if model parameters are updated or not
                assert not torch.equal(before_backward_params, after_backward_params.data)

            metrics = metric_monitor(v2lacc_r1=i2t_accs[0], l2vacc_r1=t2i_accs[0], loss=loss,
                                     use_distributed=False, metric_names=self.train_metric_names)
            train_stats.update(metric_vals=metrics, batch_time=batch_load_toc, n=batch_size)

            non_debug_total_samples = max(self.args.epochs - self.args.start_epoch + 1, 0)*len(self.train_dataloader)
            debug_total_samples = self.args.train_num_samples

            if batch_id % self.args.log_freq == 0:
                lr = self.scheduler.retrieve_lr(self.optimizer)
                train_stats.iter_summary(epoch=epoch,
                                         n_processed_samples=self.train_iterations,
                                         total_samples=debug_total_samples if self.args.debug else non_debug_total_samples,
                                         learning_rate=lr,
                                         elapsed_time=epoch_start_time)

            self.train_iterations += 1

        train_stats.epoch_summary(epoch=epoch, stage="training")
        avg_loss = train_stats.avg_statistics(metric_name='Loss')
        avg_v2laccs = self.update_dicts(self.v2lacc_func(train_stats, self.args.train_rank))
        avg_l2vaccs = self.update_dicts(self.l2vacc_func(train_stats, self.args.train_rank))
        logger.log("Train EPOCH {} -- I2T Retrieval: {:.4f} @ R1".format(epoch, avg_v2laccs['V2LAcc_R1']))
        logger.log("Train EPOCH {} -- T2I Retrieval: {:.4f} @ R1".format(epoch, avg_l2vaccs['L2VAcc_R1']))
        logger.log('Train EPOCH {} time: {:.4f}'.format(epoch, time.time() - epoch_start_time))
        return avg_v2laccs, avg_l2vaccs, avg_loss

    def val_epoch(self, epoch):
        # Achieving results for rank 1, 5, 10, 128 --> rank, metric_names
        logit_scale = self.model.module.logit_scale if hasattr(self.model, 'module') else self.model.logit_scale
        val_stats = Statistics(metric_names=self.val_metric_names, SUPPORTED_STATS=self.val_metric_names)
        if hasattr(self.model, 'module'):
            self.model.module.eval()
            if self.model.module.training:
                logger.warning('Model is in training mode. Switching to evaluation mode')
                self.model.module.eval()
        else:
            self.model.eval()
            if self.model.training:
                logger.warning('Model is in training mode. Switching to evaluation mode')
                self.model.eval()

        total_samples = len(self.val_dataloader) if not self.args.debug else self.args.val_num_samples

        full_image_latents = []
        full_text_latents = []
        with torch.no_grad():
            epoch_start_time = time.time()
            processed_samples = 0

            for batch_id, batch in enumerate(tqdm(self.val_dataloader)):
                input_img, input_text = batch[0], batch[1]
                input_img = input_img.to(self.args.device)
                input_text = {k: v.to(self.args.device, non_blocking=True) for k, v in input_text.items()}

                image_latents, text_latents = self.model(input_img, input_text, return_latent_only=True)

                logits_image = logit_scale * image_latents @ text_latents.t()
                logits_text = logit_scale * text_latents @ image_latents.t()

                loss = self.criterion(logits_image, logits_text)

                if len(image_latents.size()) == 1:
                    image_latents = image_latents.unsqueeze(0)
                if len(text_latents.size()) == 1:
                    text_latents = text_latents.unsqueeze(0)

                full_image_latents.append(image_latents.detach())
                full_text_latents.append(text_latents.detach())

                # rank serves to picking up top rank similarity accuracy
                batch_size = image_latents.size(0)

                processed_samples += batch_size

                metrics = metric_monitor(loss=loss,
                                         use_distributed=False, metric_names=self.val_metric_names)

                val_stats.update(metric_vals=metrics, batch_time=0.0, n=batch_size)

                if batch_id % self.args.log_freq == 0:
                    lr = self.scheduler.retrieve_lr(self.optimizer)
                    val_stats.iter_summary(epoch=epoch,
                                             n_processed_samples=processed_samples,
                                             total_samples=total_samples,
                                             learning_rate=lr,
                                           elapsed_time=epoch_start_time)

        full_image_latents = torch.cat(full_image_latents, dim=0)
        full_text_latents = torch.cat(full_text_latents, dim=0)

        rank = self.args.val_rank
        similarities_T2I = torch.clamp(self.model.module.logit_scale if hasattr(self.model, 'module') else self.model.logit_scale, max=torch.log(
            torch.tensor(1. / 0.01))).exp() * full_text_latents @ full_image_latents.t()
        t2i_ranks = []
        labels = np.arange(total_samples)
        t2i_ranks.extend(self.compute_rank_onehoc(similarities_T2I, labels, keep_top_k=max(rank)))
        del similarities_T2I
        torch.cuda.empty_cache()

        labels = np.arange(total_samples)
        similarities_I2T = torch.clamp(self.model.module.logit_scale if hasattr(self.model, 'module') else self.model.logit_scale, max=torch.log(
            torch.tensor(1. / 0.01))).exp() * full_image_latents @ full_text_latents.t()
        i2t_ranks = []
        i2t_ranks.extend(self.compute_rank_onehoc(similarities_I2T, labels, keep_top_k=max(rank)))
        del similarities_I2T
        torch.cuda.empty_cache()

        i2t_accs = [sum([_ < r for _ in i2t_ranks]) / len(i2t_ranks) * 100. for r in rank]
        logger.log("Val EPOCH {} -- I2T Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10, {:.4f} @ R128".format(
            epoch, i2t_accs[0], i2t_accs[1], i2t_accs[2], i2t_accs[3]))
        t2i_accs = [sum([_ < r for _ in t2i_ranks]) / len(t2i_ranks) * 100. for r in rank]
        logger.log("Val EPOCH {} -- T2I Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10, {:.4f} @ R128".format(
            epoch, t2i_accs[0], t2i_accs[1], t2i_accs[2], t2i_accs[3]))

        metrics = metric_monitor(loss=loss, v2lacc_r1=i2t_accs[0], l2vacc_r1=t2i_accs[0],
                                 v2lacc_r5=i2t_accs[1], l2vacc_r5=t2i_accs[1],
                                 v2lacc_r10=i2t_accs[2], l2vacc_r10=t2i_accs[2],
                                 v2lacc_r128=i2t_accs[3], l2vacc_r128=t2i_accs[3],
                                 use_distributed=False, metric_names=self.val_metric_names)
        val_stats.update(metric_vals=metrics, batch_time=0.0, n=len(self.val_dataloader))

        val_stats.epoch_summary(epoch=epoch, stage="Validation")
        avg_loss = val_stats.avg_statistics(metric_name='Loss')
        avg_v2laccs = self.update_dicts(self.v2lacc_func(val_stats, self.args.val_rank))
        avg_l2vaccs = self.update_dicts(self.l2vacc_func(val_stats, self.args.val_rank))
        return avg_v2laccs, avg_l2vaccs, avg_loss

    def compute_rank_onehoc(self, similarities, labels, keep_top_k=128):
        # use for one label for each instance
        def _compute_rank(_labels, _similarities, _total_captions, _keep_top_k):
            ranks = []
            for lab, sim in zip(labels, similarities):
                sims, inds = torch.topk(sim, keep_top_k)
                rank = total_captions
                for r, ind in enumerate(inds):
                    if r >= keep_top_k:
                        break
                    if ind == lab:
                        rank = r
                        break
                ranks.append(rank)
            return ranks

        total_captions = 1e8
        ranks = _compute_rank(labels, similarities, total_captions, keep_top_k)

        return ranks

    def update_dicts(self, dicts):
        dall = {}
        for d in dicts:
            dall.update(d)
        return dall


class TrainerDistill(object):
    def __init__(self, args, model, optimizer, scheduler, val_dataloader, train_dataloader, criterion, loss_scaler=None):
        super(TrainerDistill, self).__init__()
        self.args = args
        self.model = model
        self.val_dataloader = val_dataloader
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = args.device
        self.criterion = criterion
        self.loss_scaler = loss_scaler
        self.train_iterations = args.start_iteration
        logger.log(f'Training from {args.start_iteration}th iteration')
        self.tea_v2lacc_func = lambda x, y: list({ 'TeaV2LAcc_R' + str(yy): getattr(x, 'avg_statistics')(metric_name='TeaV2LAcc_R' + str(yy)) } for yy in y)
        self.tea_l2vacc_func = lambda x, y: list({ 'TeaL2VAcc_R' + str(yy): getattr(x, 'avg_statistics')(metric_name='TeaL2VAcc_R' + str(yy)) } for yy in y)
        self.stu_v2lacc_func = lambda x, y: list({ 'StuV2LAcc_R' + str(yy): getattr(x, 'avg_statistics')(metric_name='StuV2LAcc_R' + str(yy)) } for yy in y)
        self.stu_l2vacc_func = lambda x, y: list({ 'StuL2VAcc_R' + str(yy): getattr(x, 'avg_statistics')(metric_name='StuL2VAcc_R' + str(yy)) } for yy in y)

        self.train_metric_names = [args.distill_train_metric_names] if isinstance(args.distill_train_metric_names, str) else args.distill_train_metric_names
        if 'Loss' not in self.train_metric_names:
            self.train_metric_names.append('Loss')

        self.val_metric_names = [args.distill_val_metric_names] if isinstance(args.distill_val_metric_names, str) else args.distill_val_metric_names
        if 'Loss' not in self.val_metric_names:
            self.val_metric_names.append('Loss')

        self.tb_log_writter = None
        if SummaryWriter is not None:
            self.setup_log_writer()
            
        if args.amp:
            assert self.loss_scaler is not None

    def setup_log_writer(self):
        log_dir = os.path.join(self.args.output_dir, 'tb_log')
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        logger.log('Creat tb log dir!')
        self.tb_log_writter = SummaryWriter(log_dir=log_dir, comment='Training and Validation Logs')

    def run(self, train_sampler):
        if train_sampler is None:
            logger.error('Train sampler cannot be None')
        train_start_time = time.time()
        logger.log(f'Starting training from {self.args.start_epoch}th epoch')
        for epoch in range(self.args.start_epoch, self.args.epochs):
            train_sampler.set_epoch(epoch)
            # update image scales for multi-scale training
            # train_sampler.update_scales(epoch=epoch)
            tea_train_v2laccs, tea_train_l2vaccs, stu_train_v2laccs, stu_train_l2vaccs, train_loss = self.train_epoch(epoch)
            tea_val_v2laccs, tea_val_l2vaccs, stu_val_v2laccs, stu_val_l2vaccs, val_loss = self.val_epoch(epoch)

            if epoch % self.args.save_ckpt_freq == 0:
                save_ckpt_path = save_checkpoint(
                    train_iterations=self.train_iterations,
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    args=self.args,
                )
                logger.info('Checkpoints saved at: {}'.format(save_ckpt_path), print_line=True)

            if self.tb_log_writter is not None:
                lr_list = self.scheduler.retrieve_lr(self.optimizer)
                for g_id, lr_val in enumerate(lr_list):
                    self.tb_log_writter.add_scalar('LR/Group-{}'.format(g_id), round(lr_val, 6), epoch)
                self.tb_log_writter.add_scalar('Train/Loss', round(train_loss, 2), epoch)
                self.tb_log_writter.add_scalar('Val/Loss', round(val_loss, 2), epoch)
                for k, v in tea_train_v2laccs.items():
                    self.tb_log_writter.add_scalar(f'Train/{k}', round(v, 2), epoch)
                for k, v in tea_train_l2vaccs.items():
                    self.tb_log_writter.add_scalar(f'Train/{k}', round(v, 2), epoch)
                for k, v in stu_train_v2laccs.items():
                    self.tb_log_writter.add_scalar(f'Train/{k}', round(v, 2), epoch)
                for k, v in stu_train_l2vaccs.items():
                    self.tb_log_writter.add_scalar(f'Train/{k}', round(v, 2), epoch)
                for k, v in tea_val_v2laccs.items():
                    self.tb_log_writter.add_scalar(f'Val/{k}', round(v, 2), epoch)
                for k, v in tea_val_l2vaccs.items():
                    self.tb_log_writter.add_scalar(f'Val/{k}', round(v, 2), epoch)
                for k, v in stu_val_v2laccs.items():
                    self.tb_log_writter.add_scalar(f'Val/{k}', round(v, 2), epoch)
                for k, v in stu_val_l2vaccs.items():
                    self.tb_log_writter.add_scalar(f'Val/{k}', round(v, 2), epoch)

        torch.cuda.empty_cache()

        if self.tb_log_writter is not None:
            self.tb_log_writter.close()

        train_end_time = time.time()
        hours, rem = divmod(train_end_time - train_start_time, 3600)
        minutes, seconds = divmod(rem, 60)
        train_time_str = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
        logger.log('Training took {}'.format(train_time_str))

    def train_epoch(self, epoch):
        # Only achieving results for rank 1 --> rank, metric_names
        train_stats = Statistics(metric_names=self.train_metric_names, SUPPORTED_STATS=self.train_metric_names)
        if hasattr(self.model, 'module'):
            self.model.module.train()
            self.model.module.train()
        else:
            self.model.train()
            self.model.train()

        epoch_start_time = time.time()
        batch_load_start = time.time()
        for batch_id, batch in enumerate(tqdm(self.train_dataloader)):
            batch_load_toc = time.time() - batch_load_start
            input_tea_img, input_stu_img, input_text = batch[0], batch[1], batch[2]
            input_tea_img = input_tea_img.to(self.args.device, non_blocking=True)
            input_stu_img = input_stu_img.to(self.args.device, non_blocking=True)
            input_text = {k: v.to(self.args.device, non_blocking=True) for k, v in input_text.items()}

            self.optimizer = self.scheduler.distill_update_lr(self.optimizer, len(self.train_dataloader), epoch, batch_id)
            if not self.args.amp:
                logits_per_image, logits_per_text, logits_per_image_teacher, logits_per_text_teacher, distill_label_image, distill_label_text = self.model(input_tea_img, input_stu_img, input_text)
                loss = self.criterion(logits_per_image, logits_per_text, distill_label_image, distill_label_text)
            else:
                with torch.cuda.amp.autocast():
                    logits_per_image, logits_per_text, logits_per_image_teacher, logits_per_text_teacher, distill_label_image, distill_label_text = self.model(input_tea_img, input_stu_img, input_text)
                    loss = self.criterion(logits_per_image, logits_per_text, distill_label_image, distill_label_text)
            loss = loss / self.args.accumulate_step

            batch_size = logits_per_text_teacher.size(0)
            if hasattr(self.model, 'module'):
                logits_per_image, logits_per_text, logits_per_image_teacher, logits_per_text_teacher, distill_label_image, distill_label_text = self.model.module(
                    input_tea_img, input_stu_img, input_text)
            # rank serves to picking up top rank similarity accuracy
            rank = self.args.distill_train_rank
            distill_label_image, distill_label_text = np.asarray(distill_label_image.cpu()), np.asarray(
                distill_label_text.cpu())
            t2i_ranks = self.compute_rank_onehoc(logits_per_text.detach(), distill_label_text, keep_top_k=max(rank))
            i2t_ranks = self.compute_rank_onehoc(logits_per_image.detach(), distill_label_image, keep_top_k=max(rank))
            i2t_accs = [sum([_ < r for _ in i2t_ranks]) / len(i2t_ranks) * 100. for r in rank]
            t2i_accs = [sum([_ < r for _ in t2i_ranks]) / len(t2i_ranks) * 100. for r in rank]


            labels = np.arange(logits_per_text_teacher.size(0))
            t2i_ranks_teacher = self.compute_rank_onehoc(logits_per_text_teacher.detach(), labels, keep_top_k=max(rank))
            i2t_ranks_teacher = self.compute_rank_onehoc(logits_per_image_teacher.detach(), labels, keep_top_k=max(rank))
            i2t_accs_teacher = [sum([_ < r for _ in i2t_ranks_teacher]) / len(i2t_ranks_teacher) * 100. for r in rank]
            t2i_accs_teacher = [sum([_ < r for _ in t2i_ranks_teacher]) / len(t2i_ranks_teacher) * 100. for r in rank]

            if isinstance(loss, torch.Tensor) and torch.isnan(loss):
                import pdb
                pdb.set_trace()


            # before_backward_params = None
            # params_name = None
            # for idx, (name, params) in enumerate(self.model.named_parameters()):
            #     if params.requires_grad:
            #         before_backward_params = copy.deepcopy(params.data)
            #         params_name = name
            #         break

            if not self.args.amp:
                loss.backward()
            else:
                self.loss_scaler.scale(loss).backward()

            if (batch_id + 1) % self.args.accumulate_step == 0:
                if self.args.clip_grad is not None:
                    # For gradient clipping, unscale the gradients and then clip them
                    self.loss_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.clip_grad)

                if not self.args.amp:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                else:
                    self.loss_scaler.step(self.optimizer)
                    self.loss_scaler.update()
                    self.optimizer.zero_grad()

                # after_backward_params = None
                # for idx, (name, params) in enumerate(self.model.named_parameters()):
                #     if params.requires_grad and name == params_name:
                #         after_backward_params = params
                #         break

                # check if model parameters are updated or not
                # if not torch.equal(before_backward_params, after_backward_params.data):

            metrics = distill_metric_monitor(stu_v2lacc_r1=i2t_accs[0], stu_l2vacc_r1=t2i_accs[0],
                                             tea_v2lacc_r1=i2t_accs_teacher[0], tea_l2vacc_r1=t2i_accs_teacher[0],
                                             loss=loss, use_distributed=False, metric_names=self.train_metric_names)
            train_stats.update(metric_vals=metrics, batch_time=batch_load_toc, n=logits_per_image.size(0))

            non_debug_total_samples = max(self.args.epochs - self.args.start_epoch + 1, 0)*len(self.train_dataloader)
            debug_total_samples = self.args.train_num_samples

            if batch_id % self.args.log_freq == 0:
                lr = self.scheduler.retrieve_lr(self.optimizer)
                train_stats.iter_summary(epoch=epoch,
                                         n_processed_samples=self.train_iterations,
                                         total_samples=debug_total_samples if self.args.debug else non_debug_total_samples, # !Wrongly, it needs to be modified!
                                         learning_rate=lr,
                                         elapsed_time=epoch_start_time)

            self.train_iterations += 1

        train_stats.epoch_summary(epoch=epoch, stage="Training")
        avg_loss = train_stats.avg_statistics(metric_name='Loss')
        tea_avg_v2laccs = self.update_dicts(self.tea_v2lacc_func(train_stats, self.args.distill_train_rank))
        tea_avg_l2vaccs = self.update_dicts(self.tea_l2vacc_func(train_stats, self.args.distill_train_rank))
        stu_avg_v2laccs = self.update_dicts(self.stu_v2lacc_func(train_stats, self.args.distill_train_rank))
        stu_avg_l2vaccs = self.update_dicts(self.stu_l2vacc_func(train_stats, self.args.distill_train_rank))
        logger.log("Train EPOCH {} -- Retrieval Loss: {:.4f}".format(epoch, avg_loss))
        logger.log("Teacher Train EPOCH {} -- I2T Retrieval: {:.4f} @ R1".format(epoch, tea_avg_v2laccs['TeaV2LAcc_R1']))
        logger.log("Teacher Train EPOCH {} -- T2I Retrieval: {:.4f} @ R1".format(epoch, tea_avg_l2vaccs['TeaL2VAcc_R1']))
        logger.log("Student Train EPOCH {} -- I2T Retrieval: {:.4f} @ R1".format(epoch, stu_avg_v2laccs['StuV2LAcc_R1']))
        logger.log("Student Train EPOCH {} -- T2I Retrieval: {:.4f} @ R1".format(epoch, stu_avg_l2vaccs['StuL2VAcc_R1']))
        logger.log('Train EPOCH {} time: {:.4f}'.format(epoch, time.time() - epoch_start_time))
        return tea_avg_v2laccs, tea_avg_l2vaccs, stu_avg_v2laccs, stu_avg_l2vaccs, avg_loss

    def val_epoch(self, epoch):
        logit_scale = self.model.module.logit_scale if hasattr(self.model, 'module') else self.model.logit_scale
        # Achieving results for rank 1, 5, 10, 128 --> rank, metric_names
        val_stats = Statistics(metric_names=self.val_metric_names, SUPPORTED_STATS=self.val_metric_names)
        if hasattr(self.model, 'module'):
            self.model.module.eval()
        else:
            self.model.eval()

        if hasattr(self.model, 'module'):
            if self.model.module.training:
                logger.warning('Model is in training mode. Switching to evaluation mode')
                self.model.module.eval()
        else:
            if self.model.training:
                logger.warning('Model is in training mode. Switching to evaluation mode')
                self.model.eval()

        total_samples = len(self.val_dataloader) if not self.args.debug else self.args.val_num_samples

        with torch.no_grad():
            epoch_start_time = time.time()
            processed_samples = 0

            for batch_id, batch in enumerate(tqdm(self.val_dataloader)):
                input_tea_img, input_stu_img, input_text = batch[0], batch[1], batch[2]
                input_tea_img = input_tea_img.to(self.args.device)
                input_stu_img = input_stu_img.to(self.args.device)
                input_text = {k: v.to(self.args.device, non_blocking=True) for k, v in input_text.items()}

                if hasattr(self.model, 'module'):
                    tea_img_latents, stu_img_latents, text_latents = self.model.module(
                        input_tea_img, input_stu_img, input_text, return_latent_only=True)
                else:
                    tea_img_latents, stu_img_latents, text_latents = self.model(
                        input_tea_img, input_stu_img, input_text, return_latent_only=True)

                logits_per_image = logit_scale * stu_img_latents @ text_latents.t()
                logits_per_text = logit_scale * text_latents @ stu_img_latents.t()
                logits_per_image_teacher = tea_img_latents @ text_latents.t()
                logits_per_text_teacher = text_latents @ tea_img_latents.t()
                distill_label_image = torch.argmax(logits_per_image_teacher, dim=-1)
                distill_label_text = torch.argmax(logits_per_text_teacher, dim=-1)

                loss = self.criterion(logits_per_image, logits_per_text, distill_label_image, distill_label_text)

                # rank serves to picking up top rank similarity accuracy
                batch_size = tea_img_latents.size(0)

                processed_samples += batch_size

                #########################################################################
                rank = self.args.distill_val_rank
                t2i_ranks = self.compute_rank_onehoc(logits_per_text.detach(), distill_label_text, keep_top_k=max(rank))
                i2t_ranks = self.compute_rank_onehoc(logits_per_image.detach(), distill_label_image, keep_top_k=max(rank))
                i2t_accs_stu = [sum([_ < r for _ in i2t_ranks]) / len(i2t_ranks) * 100. for r in rank]
                t2i_accs_stu = [sum([_ < r for _ in t2i_ranks]) / len(t2i_ranks) * 100. for r in rank]

                labels = np.arange(logits_per_text_teacher.size(0))
                t2i_ranks_teacher = self.compute_rank_onehoc(logits_per_text_teacher.detach(), labels, keep_top_k=max(rank))
                i2t_ranks_teacher = self.compute_rank_onehoc(logits_per_image_teacher.detach(), labels, keep_top_k=max(rank))
                i2t_accs_teacher = [sum([_ < r for _ in i2t_ranks_teacher]) / len(i2t_ranks_teacher) * 100. for r in rank]
                t2i_accs_teacher = [sum([_ < r for _ in t2i_ranks_teacher]) / len(t2i_ranks_teacher) * 100. for r in rank]

                metrics = distill_metric_monitor(loss=loss, tea_v2lacc_r1=i2t_accs_teacher[0], tea_l2vacc_r1=t2i_accs_teacher[0],
                                                 tea_v2lacc_r5=i2t_accs_teacher[1], tea_l2vacc_r5=t2i_accs_teacher[1],
                                                 tea_v2lacc_r10=i2t_accs_teacher[2], tea_l2vacc_r10=t2i_accs_teacher[2],
                                                 # tea_v2lacc_r128=i2t_accs_teacher[3], tea_l2vacc_r128=t2i_accs_teacher[3],
                                                 stu_v2lacc_r1=i2t_accs_stu[0], stu_l2vacc_r1=t2i_accs_stu[0],
                                                 stu_v2lacc_r5=i2t_accs_stu[1], stu_l2vacc_r5=t2i_accs_stu[1],
                                                 stu_v2lacc_r10=i2t_accs_stu[2], stu_l2vacc_r10=t2i_accs_stu[2],
                                                 # stu_v2lacc_r128=i2t_accs_stu[3], stu_l2vacc_r128=t2i_accs_stu[3],
                                                 use_distributed=False, metric_names=self.val_metric_names)
                val_stats.update(metric_vals=metrics, batch_time=0.0, n=batch_size)

                if batch_id % self.args.log_freq == 0:
                    lr = self.scheduler.retrieve_lr(self.optimizer)
                    val_stats.iter_summary(epoch=epoch,
                                           n_processed_samples=processed_samples,
                                           total_samples=total_samples,   # Wrongly for computing processed_samples and total_samples
                                           learning_rate=lr,
                                           elapsed_time=epoch_start_time)


        val_stats.epoch_summary(epoch=epoch, stage="Validation")
        avg_loss = val_stats.avg_statistics(metric_name='Loss')
        logger.log("Val EPOCH {} -- Retrieval Loss: {:.4f}".format(epoch, avg_loss))
        tea_avg_v2laccs = self.update_dicts(self.tea_v2lacc_func(val_stats, self.args.distill_val_rank))
        tea_avg_l2vaccs = self.update_dicts(self.tea_l2vacc_func(val_stats, self.args.distill_val_rank))
        stu_avg_v2laccs = self.update_dicts(self.stu_v2lacc_func(val_stats, self.args.distill_val_rank))
        stu_avg_l2vaccs = self.update_dicts(self.stu_l2vacc_func(val_stats, self.args.distill_val_rank))
        logger.log("Teacher Val EPOCH {} -- I2T Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(
            epoch, tea_avg_v2laccs['TeaV2LAcc_R1'], tea_avg_v2laccs['TeaV2LAcc_R5'], tea_avg_v2laccs['TeaV2LAcc_R10']))
        logger.log("Teacher Val EPOCH {} -- T2I Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(
            epoch, tea_avg_l2vaccs['TeaL2VAcc_R1'], tea_avg_l2vaccs['TeaL2VAcc_R5'], tea_avg_l2vaccs['TeaL2VAcc_R10']))

        logger.log("Student Val EPOCH {} -- I2T Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(
            epoch, stu_avg_v2laccs['StuV2LAcc_R1'], stu_avg_v2laccs['StuV2LAcc_R5'], stu_avg_v2laccs['StuV2LAcc_R10']))
        logger.log("Student Val EPOCH {} -- T2I Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(
            epoch, stu_avg_l2vaccs['StuL2VAcc_R1'], stu_avg_l2vaccs['StuL2VAcc_R5'], stu_avg_l2vaccs['StuL2VAcc_R10']))
        return tea_avg_v2laccs, tea_avg_l2vaccs, stu_avg_v2laccs, stu_avg_l2vaccs, avg_loss

    def compute_rank_onehoc(self, similarities, labels, keep_top_k=128):
        # use for one label for each instance
        def _compute_rank(_labels, _similarities, _total_captions, _keep_top_k):
            ranks = []
            for lab, sim in zip(labels, similarities):
                sims, inds = torch.topk(sim, keep_top_k)
                rank = total_captions
                for r, ind in enumerate(inds):
                    if r >= keep_top_k:
                        break
                    if ind == lab:
                        rank = r
                        break
                ranks.append(rank)
            return ranks

        total_captions = 1e8
        ranks = _compute_rank(labels, similarities, total_captions, keep_top_k)

        return ranks

    def update_dicts(self, dicts):
        dall = {}
        for d in dicts:
            dall.update(d)
        return dall


class TrainerDistributedDistill(object):
    def __init__(self, args, model, optimizer, scheduler, val_dataloader, train_dataloader, criterion, is_mater, loss_scaler=None):
        super(TrainerDistributedDistill, self).__init__()
        self.args = args
        self.model = model
        self.val_dataloader = val_dataloader
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = args.device
        self.criterion = criterion
        self.train_iterations = args.start_iteration
        self.is_master = is_mater
        self.loss_scaler = loss_scaler
        logger.log(f'Training from {args.start_iteration}th iteration')
        self.tea_v2lacc_func = lambda x, y: list({ 'TeaV2LAcc_R' + str(yy): getattr(x, 'avg_statistics')(metric_name='TeaV2LAcc_R' + str(yy)) } for yy in y)
        self.tea_l2vacc_func = lambda x, y: list({ 'TeaL2VAcc_R' + str(yy): getattr(x, 'avg_statistics')(metric_name='TeaL2VAcc_R' + str(yy)) } for yy in y)
        self.stu_v2lacc_func = lambda x, y: list({ 'StuV2LAcc_R' + str(yy): getattr(x, 'avg_statistics')(metric_name='StuV2LAcc_R' + str(yy)) } for yy in y)
        self.stu_l2vacc_func = lambda x, y: list({ 'StuL2VAcc_R' + str(yy): getattr(x, 'avg_statistics')(metric_name='StuL2VAcc_R' + str(yy)) } for yy in y)

        self.train_metric_names = [args.distill_train_metric_names] if isinstance(args.distill_train_metric_names, str) else args.distill_train_metric_names
        if 'Loss' not in self.train_metric_names:
            self.train_metric_names.append('Loss')

        self.val_metric_names = [args.distill_val_metric_names] if isinstance(args.distill_val_metric_names, str) else args.distill_val_metric_names
        if 'Loss' not in self.val_metric_names:
            self.val_metric_names.append('Loss')

        self.tb_log_writter = None
        if SummaryWriter is not None and self.is_master:
            self.setup_log_writer()
            
        if args.amp:
            assert self.loss_scaler is not None

    def setup_log_writer(self):
        log_dir = os.path.join(self.args.output_dir, 'tb_log')
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        if self.is_master:
            logger.log('Creat tb log dir!')
        self.tb_log_writter = SummaryWriter(log_dir=log_dir, comment='Training and Validation Logs')

    def run(self, train_sampler):
        if train_sampler is None and self.is_master:
            logger.error('Train sampler cannot be None')

        train_start_time = time.time()
        if self.is_master:
            logger.log(f'Starting training from {self.args.start_epoch}th epoch')
        for epoch in range(self.args.start_epoch, self.args.epochs):
            train_sampler.set_epoch(epoch)
            # update image scales for multi-scale training
            # train_sampler.update_scales(epoch=epoch)
            tea_train_v2laccs, tea_train_l2vaccs, stu_train_v2laccs, stu_train_l2vaccs, train_loss = self.train_epoch(epoch)

            if epoch % self.args.save_ckpt_freq == 0 and self.is_master:
                save_ckpt_path = save_checkpoint(
                    train_iterations=self.train_iterations,
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    args=self.args,
                )
                logger.info('Checkpoints saved at: {}'.format(save_ckpt_path), print_line=True)


            tea_val_v2laccs, tea_val_l2vaccs, stu_val_v2laccs, stu_val_l2vaccs, val_loss = self.val_epoch(epoch)

            if self.tb_log_writter is not None and self.is_master:
                lr_list = self.scheduler.retrieve_lr(self.optimizer)
                for g_id, lr_val in enumerate(lr_list):
                    self.tb_log_writter.add_scalar('LR/Group-{}'.format(g_id), round(lr_val, 6), epoch)
                self.tb_log_writter.add_scalar('Train/Loss', round(train_loss, 2), epoch)
                self.tb_log_writter.add_scalar('Val/Loss', round(val_loss, 2), epoch)
                for k, v in tea_train_v2laccs.items():
                    self.tb_log_writter.add_scalar(f'Train/{k}', round(v, 2), epoch)
                for k, v in tea_train_l2vaccs.items():
                    self.tb_log_writter.add_scalar(f'Train/{k}', round(v, 2), epoch)
                for k, v in stu_train_v2laccs.items():
                    self.tb_log_writter.add_scalar(f'Train/{k}', round(v, 2), epoch)
                for k, v in stu_train_l2vaccs.items():
                    self.tb_log_writter.add_scalar(f'Train/{k}', round(v, 2), epoch)
                for k, v in tea_val_v2laccs.items():
                    self.tb_log_writter.add_scalar(f'Val/{k}', round(v, 2), epoch)
                for k, v in tea_val_l2vaccs.items():
                    self.tb_log_writter.add_scalar(f'Val/{k}', round(v, 2), epoch)
                for k, v in stu_val_v2laccs.items():
                    self.tb_log_writter.add_scalar(f'Val/{k}', round(v, 2), epoch)
                for k, v in stu_val_l2vaccs.items():
                    self.tb_log_writter.add_scalar(f'Val/{k}', round(v, 2), epoch)

        torch.cuda.empty_cache()

        if self.tb_log_writter is not None and self.is_master:
            self.tb_log_writter.close()

        train_end_time = time.time()
        hours, rem = divmod(train_end_time - train_start_time, 3600)
        minutes, seconds = divmod(rem, 60)
        train_time_str = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
        logger.log('Training took {}'.format(train_time_str))

    def train_epoch(self, epoch):
        # Only achieving results for rank 1 --> rank, metric_names
        train_stats = Statistics(metric_names=self.train_metric_names, SUPPORTED_STATS=self.train_metric_names)
        self.model.train()

        epoch_start_time = time.time()
        batch_load_start = time.time()
        for batch_id, batch in enumerate(tqdm(self.train_dataloader)):
            batch_load_toc = time.time() - batch_load_start
            input_tea_img, input_stu_img, input_text = batch[0], batch[1], batch[2]
            input_tea_img = input_tea_img.to(self.args.device, non_blocking=True)
            input_stu_img = input_stu_img.to(self.args.device, non_blocking=True)
            input_text = {k: v.to(self.args.device, non_blocking=True) for k, v in input_text.items()}

            self.optimizer = self.scheduler.distill_update_lr(self.optimizer, len(self.train_dataloader), epoch, batch_id)
            if not self.args.amp:
                logits_per_image, logits_per_text, logits_per_image_teacher, logits_per_text_teacher, distill_label_image, distill_label_text = self.model(input_tea_img, input_stu_img, input_text)
                loss = self.criterion(logits_per_image, logits_per_text, distill_label_image, distill_label_text)
            else:
                with torch.cuda.amp.autocast():
                    logits_per_image, logits_per_text, logits_per_image_teacher, logits_per_text_teacher, distill_label_image, distill_label_text = self.model(input_tea_img, input_stu_img, input_text)
                    loss = self.criterion(logits_per_image, logits_per_text, distill_label_image, distill_label_text)
                
            loss = loss / self.args.accumulate_step

            batch_size = logits_per_text_teacher.size(0)
            # rank serves to picking up top rank similarity accuracy
            rank = self.args.distill_train_rank
            distill_label_image, distill_label_text = np.asarray(distill_label_image.cpu()), np.asarray(
                distill_label_text.cpu())
            t2i_ranks = self.compute_rank_onehoc(logits_per_text.detach(), distill_label_text, keep_top_k=max(rank))
            i2t_ranks = self.compute_rank_onehoc(logits_per_image.detach(), distill_label_image, keep_top_k=max(rank))
            i2t_accs = [sum([_ < r for _ in i2t_ranks]) / len(i2t_ranks) * 100. for r in rank]
            t2i_accs = [sum([_ < r for _ in t2i_ranks]) / len(t2i_ranks) * 100. for r in rank]


            labels = np.arange(logits_per_text_teacher.size(0))
            t2i_ranks_teacher = self.compute_rank_onehoc(logits_per_text_teacher.detach(), labels, keep_top_k=max(rank))
            i2t_ranks_teacher = self.compute_rank_onehoc(logits_per_image_teacher.detach(), labels, keep_top_k=max(rank))
            i2t_accs_teacher = [sum([_ < r for _ in i2t_ranks_teacher]) / len(i2t_ranks_teacher) * 100. for r in rank]
            t2i_accs_teacher = [sum([_ < r for _ in t2i_ranks_teacher]) / len(t2i_ranks_teacher) * 100. for r in rank]

            if isinstance(loss, torch.Tensor) and torch.isnan(loss):
                import pdb
                pdb.set_trace()

            # before_backward_params = None
            # params_name = None
            # for idx, (name, params) in enumerate(self.model.named_parameters()):
            #     if params.requires_grad:
            #         before_backward_params = copy.deepcopy(params.data)
            #         params_name = name
            #         break

            if not self.args.amp:            
                loss.backward()
            else:
                self.loss_scaler.scale(loss).backward()

            if (batch_id + 1) % self.args.accumulate_step == 0:
                if self.args.clip_grad is not None:
                    # For gradient clipping, unscale the gradients and then clip them
                    # self.gradient_scalar.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.clip_grad)

                if not self.args.amp:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                else:
                    self.loss_scaler.step(self.optimizer)
                    self.loss_scaler.update()
                    self.optimizer.zero_grad()

                # after_backward_params = None
                # for idx, (name, params) in enumerate(self.model.named_parameters()):
                #     if params.requires_grad and name == params_name:
                #         after_backward_params = params
                #         break
                #
                # # check if model parameters are updated or not
                # assert not torch.equal(before_backward_params, after_backward_params.data), params_name

            metrics = distill_metric_monitor(stu_v2lacc_r1=i2t_accs[0], stu_l2vacc_r1=t2i_accs[0],
                                             tea_v2lacc_r1=i2t_accs_teacher[0], tea_l2vacc_r1=t2i_accs_teacher[0],
                                             loss=loss.item(), use_distributed=True, metric_names=self.train_metric_names)
            train_stats.update(metric_vals=metrics, batch_time=batch_load_toc, n=logits_per_image.size(0))

            non_debug_total_samples = max(self.args.epochs - self.args.start_epoch + 1, 0)*len(self.train_dataloader)
            debug_total_samples = self.args.train_num_samples

            if batch_id % self.args.log_freq == 0:
                lr = self.scheduler.retrieve_lr(self.optimizer)
                train_stats.iter_summary(epoch=epoch,
                                         n_processed_samples=self.train_iterations,
                                         total_samples=debug_total_samples if self.args.debug else non_debug_total_samples, # !Wrongly, it needs to be modified!
                                         learning_rate=lr,
                                         elapsed_time=epoch_start_time)

            self.train_iterations += 1
            torch.cuda.synchronize()

        train_stats.epoch_summary(epoch=epoch, stage="Training")
        avg_loss = train_stats.avg_statistics(metric_name='Loss')
        tea_avg_v2laccs = self.update_dicts(self.tea_v2lacc_func(train_stats, self.args.distill_train_rank))
        tea_avg_l2vaccs = self.update_dicts(self.tea_l2vacc_func(train_stats, self.args.distill_train_rank))
        stu_avg_v2laccs = self.update_dicts(self.stu_v2lacc_func(train_stats, self.args.distill_train_rank))
        stu_avg_l2vaccs = self.update_dicts(self.stu_l2vacc_func(train_stats, self.args.distill_train_rank))

        logger.log("Train EPOCH {} -- Retrieval Loss: {:.4f}".format(epoch, avg_loss))
        logger.log("Teacher Train EPOCH {} -- I2T Retrieval: {:.4f} @ R1".format(epoch, tea_avg_v2laccs['TeaV2LAcc_R1']))
        logger.log("Teacher Train EPOCH {} -- T2I Retrieval: {:.4f} @ R1".format(epoch, tea_avg_l2vaccs['TeaL2VAcc_R1']))
        logger.log("Student Train EPOCH {} -- I2T Retrieval: {:.4f} @ R1".format(epoch, stu_avg_v2laccs['StuV2LAcc_R1']))
        logger.log("Student Train EPOCH {} -- T2I Retrieval: {:.4f} @ R1".format(epoch, stu_avg_l2vaccs['StuL2VAcc_R1']))
        logger.log('Train EPOCH {} time: {:.4f}'.format(epoch, time.time() - epoch_start_time))
        return tea_avg_v2laccs, tea_avg_l2vaccs, stu_avg_v2laccs, stu_avg_l2vaccs, avg_loss

    def val_epoch(self, epoch):
        logit_scale = self.model.module.logit_scale if hasattr(self.model, 'module') else self.model.logit_scale
        # Achieving results for rank 1, 5, 10, 128 --> rank, metric_names
        val_stats = Statistics(metric_names=self.val_metric_names, SUPPORTED_STATS=self.val_metric_names)
        self.model.eval()

        if self.model.training:
            logger.warning('Model is in training mode. Switching to evaluation mode')
            self.model.eval()

        total_samples = len(self.val_dataloader) if not self.args.debug else self.args.val_num_samples

        with torch.no_grad():
            epoch_start_time = time.time()
            processed_samples = 0

            for batch_id, batch in enumerate(tqdm(self.val_dataloader)):
                input_tea_img, input_stu_img, input_text = batch[0], batch[1], batch[2]
                input_tea_img = input_tea_img.to(self.args.device)
                input_stu_img = input_stu_img.to(self.args.device)
                input_text = {k: v.to(self.args.device, non_blocking=True) for k, v in input_text.items()}

                tea_img_latents, stu_img_latents, text_latents = self.model(
                input_tea_img, input_stu_img, input_text, return_latent_only=True)

                logits_per_image = logit_scale * stu_img_latents @ text_latents.t()
                logits_per_text = logit_scale * text_latents @ stu_img_latents.t()
                logits_per_image_teacher = tea_img_latents @ text_latents.t()
                logits_per_text_teacher = text_latents @ tea_img_latents.t()
                distill_label_image = torch.argmax(logits_per_image_teacher, dim=-1)
                distill_label_text = torch.argmax(logits_per_text_teacher, dim=-1)

                loss = self.criterion(logits_per_image, logits_per_text, distill_label_image, distill_label_text)

                # rank serves to picking up top rank similarity accuracy
                batch_size = tea_img_latents.size(0)

                processed_samples += batch_size

                #########################################################################
                rank = self.args.distill_val_rank
                t2i_ranks = self.compute_rank_onehoc(logits_per_text.detach(), distill_label_text, keep_top_k=max(rank))
                i2t_ranks = self.compute_rank_onehoc(logits_per_image.detach(), distill_label_image, keep_top_k=max(rank))
                i2t_accs_stu = [sum([_ < r for _ in i2t_ranks]) / len(i2t_ranks) * 100. for r in rank]
                t2i_accs_stu = [sum([_ < r for _ in t2i_ranks]) / len(t2i_ranks) * 100. for r in rank]

                labels = np.arange(logits_per_text_teacher.size(0))
                t2i_ranks_teacher = self.compute_rank_onehoc(logits_per_text_teacher.detach(), labels, keep_top_k=max(rank))
                i2t_ranks_teacher = self.compute_rank_onehoc(logits_per_image_teacher.detach(), labels, keep_top_k=max(rank))
                i2t_accs_teacher = [sum([_ < r for _ in i2t_ranks_teacher]) / len(i2t_ranks_teacher) * 100. for r in rank]
                t2i_accs_teacher = [sum([_ < r for _ in t2i_ranks_teacher]) / len(t2i_ranks_teacher) * 100. for r in rank]

                metrics = distill_metric_monitor(loss=loss, tea_v2lacc_r1=i2t_accs_teacher[0], tea_l2vacc_r1=t2i_accs_teacher[0],
                                                 tea_v2lacc_r5=i2t_accs_teacher[1], tea_l2vacc_r5=t2i_accs_teacher[1],
                                                 tea_v2lacc_r10=i2t_accs_teacher[2], tea_l2vacc_r10=t2i_accs_teacher[2],
                                                 # tea_v2lacc_r128=i2t_accs_teacher[3], tea_l2vacc_r128=t2i_accs_teacher[3],
                                                 stu_v2lacc_r1=i2t_accs_stu[0], stu_l2vacc_r1=t2i_accs_stu[0],
                                                 stu_v2lacc_r5=i2t_accs_stu[1], stu_l2vacc_r5=t2i_accs_stu[1],
                                                 stu_v2lacc_r10=i2t_accs_stu[2], stu_l2vacc_r10=t2i_accs_stu[2],
                                                 # stu_v2lacc_r128=i2t_accs_stu[3], stu_l2vacc_r128=t2i_accs_stu[3],
                                                 use_distributed=True, metric_names=self.val_metric_names)
                val_stats.update(metric_vals=metrics, batch_time=0.0, n=batch_size)

                if batch_id % self.args.log_freq == 0:
                    lr = self.scheduler.retrieve_lr(self.optimizer)
                    val_stats.iter_summary(epoch=epoch,
                                           n_processed_samples=processed_samples,
                                           total_samples=total_samples,   # Wrongly for computing processed_samples and total_samples
                                           learning_rate=lr,
                                           elapsed_time=epoch_start_time)

        val_stats.epoch_summary(epoch=epoch, stage="Validation")
        avg_loss = val_stats.avg_statistics(metric_name='Loss')
        logger.log("Val EPOCH {} -- Retrieval Loss: {:.4f}".format(epoch, avg_loss))
        tea_avg_v2laccs = self.update_dicts(self.tea_v2lacc_func(val_stats, self.args.distill_val_rank))
        tea_avg_l2vaccs = self.update_dicts(self.tea_l2vacc_func(val_stats, self.args.distill_val_rank))
        stu_avg_v2laccs = self.update_dicts(self.stu_v2lacc_func(val_stats, self.args.distill_val_rank))
        stu_avg_l2vaccs = self.update_dicts(self.stu_l2vacc_func(val_stats, self.args.distill_val_rank))
        logger.log("Teacher Val EPOCH {} -- I2T Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(
            epoch, tea_avg_v2laccs['TeaV2LAcc_R1'], tea_avg_v2laccs['TeaV2LAcc_R5'], tea_avg_v2laccs['TeaV2LAcc_R10']))
        logger.log("Teacher Val EPOCH {} -- T2I Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(
            epoch, tea_avg_l2vaccs['TeaL2VAcc_R1'], tea_avg_l2vaccs['TeaL2VAcc_R5'], tea_avg_l2vaccs['TeaL2VAcc_R10']))

        logger.log("Student Val EPOCH {} -- I2T Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(
            epoch, stu_avg_v2laccs['StuV2LAcc_R1'], stu_avg_v2laccs['StuV2LAcc_R5'], stu_avg_v2laccs['StuV2LAcc_R10']))
        logger.log("Student Val EPOCH {} -- T2I Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(
            epoch, stu_avg_l2vaccs['StuL2VAcc_R1'], stu_avg_l2vaccs['StuL2VAcc_R5'], stu_avg_l2vaccs['StuL2VAcc_R10']))
        return tea_avg_v2laccs, tea_avg_l2vaccs, stu_avg_v2laccs, stu_avg_l2vaccs, avg_loss

    def compute_rank_onehoc(self, similarities, labels, keep_top_k=128):
        # use for one label for each instance
        def _compute_rank(_labels, _similarities, _total_captions, _keep_top_k):
            ranks = []
            for lab, sim in zip(labels, similarities):
                sims, inds = torch.topk(sim, keep_top_k)
                rank = total_captions
                for r, ind in enumerate(inds):
                    if r >= keep_top_k:
                        break
                    if ind == lab:
                        rank = r
                        break
                ranks.append(rank)
            return ranks

        total_captions = 1e8
        ranks = _compute_rank(labels, similarities, total_captions, keep_top_k)

        return ranks

    def update_dicts(self, dicts):
        dall = {}
        for d in dicts:
            dall.update(d)
        return dall
