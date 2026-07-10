import logging

import numpy as np
import torch
from time import time
from torch import optim
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

from utils import ensure_dir,set_color,get_local_time,delete_file
import os
import json

import heapq
class Trainer(object):

    def __init__(self, args, model, data_num):
        self.args = args
        self.model = model
        self.logger = logging.getLogger()

        self.lr = args.lr
        self.learner = args.learner
        self.lr_scheduler_type = args.lr_scheduler_type

        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.warmup_steps = args.warmup_epochs * data_num
        self.max_steps = args.epochs * data_num

        self.save_limit = args.save_limit
        self.best_save_heap = []
        self.newest_save_queue = []
        self.eval_step = min(args.eval_step, self.epochs)
        self.device = args.device
        self.device = torch.device(self.device)
        self.ckpt_dir = args.ckpt_dir
        saved_model_dir = "{}".format(get_local_time())
        self.ckpt_dir = os.path.join(self.ckpt_dir,saved_model_dir)
        ensure_dir(self.ckpt_dir)

        self.best_loss = np.inf
        self.best_collision_rate = np.inf
        self.best_loss_ckpt = "best_loss_model.pth"
        self.best_collision_ckpt = "best_collision_model.pth"

        # 结构化 metrics 日志：每个 eval_step 记录一次，训练结束后保存为 JSON
        self.metrics_log = []
        self._metrics_json_path = os.path.join(self.ckpt_dir, "training_metrics.json")

        self.optimizer = self._build_optimizer()
        self.scheduler = self._get_scheduler()
        self.model = self.model.to(self.device)

    def _collect_metrics(self, epoch_idx, train_loss, train_recon_loss, collision_rate):
        """
        收集当前 eval_step 的所有指标，追加到结构化日志中。

        指标包括：
          - epoch: 当前 epoch
          - train_loss / train_recon_loss: 训练损失
          - collision_rate: SID 碰撞率
          - codebook_usage: 每层 codebook 的 perplexity、dead_code 等
        """
        record = {
            'epoch': epoch_idx,
            'train_loss': round(train_loss, 6),
            'train_recon_loss': round(train_recon_loss, 6),
            'collision_rate': round(collision_rate, 6),
        }

        # 收集 codebook usage（如果模型支持）
        try:
            usage = self.model.rq.get_codebook_usage()
            summary = usage['summary']
            record['codebook'] = {
                'avg_perplexity': round(summary['perplexity_mean'], 2),
                'total_dead_codes': summary['dead_count_total'],
                'dead_code_ratio': round(summary['dead_ratio_mean'], 4),
                'per_layer': {}
            }
            for i in range(len(usage) - 1):  # exclude 'summary' key
                layer_key = f'layer_{i}'
                if layer_key in usage:
                    u = usage[layer_key]
                    record['codebook']['per_layer'][layer_key] = {
                        'perplexity': round(u['perplexity'], 2),
                        'codebook_size': self.model.rq.n_e_list[i],
                        'dead_count': u['dead_count'],
                        'usage_min': round(u['usage_min'], 2),
                        'usage_mean': round(u['usage_mean'], 2),
                        'usage_max': round(u['usage_max'], 2),
                    }
        except Exception:
            pass  # 模型不支持 codebook usage 时静默跳过

        self.metrics_log.append(record)

        # 实时写入 JSON（确保即使训练中断也有记录）
        try:
            with open(self._metrics_json_path, 'w') as f:
                json.dump(self.metrics_log, f, indent=2)
        except Exception:
            pass

    def _log_codebook_stats(self, epoch_idx):
        """
        Log per-layer codebook utilization and gate stats at the end of each epoch.

        Prints:
          - Per-layer: perplexity / codebook_size (utilization %), dead code count
          - Collab gate stats (mean/max activation) if using fusion model
        """
        try:
            usage = self.model.rq.get_codebook_usage()
            summary = usage['summary']
        except Exception:
            return  # model doesn't support codebook usage

        # Build per-layer utilization string
        layer_parts = []
        for i in range(len(usage) - 1):  # exclude 'summary' key
            layer_key = f'layer_{i}'
            if layer_key in usage:
                u = usage[layer_key]
                ppl = u['perplexity']
                cb_size = u['codebook_size']
                util_pct = ppl / cb_size * 100
                dead = u['dead_count']
                layer_parts.append(f"L{i}: {util_pct:.0f}% ({int(ppl)}/{cb_size}) dead={dead}")

        layer_str = " | ".join(layer_parts)

        # Gate stats (if using collab fusion)
        gate_str = ""
        try:
            gate_stats = self.model.get_gate_stats()
            if gate_stats is not None:
                gate_str = (f" | Gate: mean={gate_stats['gate_mean']:.3f} "
                            f"max={gate_stats['gate_max']:.3f}")
        except Exception:
            pass

        self.logger.info(
            set_color(f"Epoch {epoch_idx} codebook", "cyan") +
            f": {layer_str}"
            f" | avg_ppl={summary['perplexity_mean']:.1f}"
            f" | total_dead={summary['dead_count_total']}"
            f"{gate_str}"
        )

    def _build_optimizer(self):

        params = self.model.parameters()
        learner =  self.learner
        learning_rate = self.lr
        weight_decay = self.weight_decay

        if learner.lower() == "adam":
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(
                params, lr=learning_rate, weight_decay=weight_decay
            )
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == 'adamw':
            optimizer = optim.AdamW(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer

    def _get_scheduler(self):
        if self.lr_scheduler_type.lower() == "linear":
            lr_scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                           num_warmup_steps=self.warmup_steps,
                                                           num_training_steps=self.max_steps)
        else:
            lr_scheduler = get_constant_schedule_with_warmup(optimizer=self.optimizer,
                                                             num_warmup_steps=self.warmup_steps)

        return lr_scheduler
    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")


    def _train_epoch(self, train_data, epoch_idx):

        self.model.train()

        total_loss = 0
        total_recon_loss = 0
        iter_data = tqdm(
                    train_data,
                    total=len(train_data),
                    ncols=120,
                    desc=set_color(f"Train {epoch_idx}","pink"),
                    )

        for batch_idx, data in enumerate(iter_data):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            out, rq_loss, indices = self.model(data)
            loss, loss_recon = self.model.compute_loss(out, rq_loss, xs=data)
            self._check_nan(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
            total_recon_loss += loss_recon.item()

            # Log codebook usage stats in tqdm postfix (first batch of each epoch)
            if batch_idx == 0:
                usage = self.model.rq.get_codebook_usage()
                summary = usage['summary']
                iter_data.set_postfix({
                    'ppl': f"{summary['perplexity_mean']:.1f}",
                    'dead': f"{summary['dead_count_total']}",
                })

        return total_loss, total_recon_loss

    @torch.no_grad()
    def _valid_epoch(self, valid_data):

        self.model.eval()

        iter_data =tqdm(
                valid_data,
                total=len(valid_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )

        indices_set = set()
        num_sample = 0
        for batch_idx, data in enumerate(iter_data):
            num_sample += len(data)
            data = data.to(self.device)
            indices = self.model.get_indices(data)
            indices = indices.view(-1,indices.shape[-1]).cpu().numpy()
            for index in indices:
                code = "-".join([str(int(_)) for _ in index])
                indices_set.add(code)

        collision_rate = (num_sample - len(list(indices_set)))/num_sample

        # Log per-level codebook usage stats
        try:
            usage = self.model.rq.get_codebook_usage()
            summary = usage['summary']
            self.logger.info(
                set_color("  Codebook Usage", "cyan") +
                f": avg_ppl={summary['perplexity_mean']:.1f} | "
                f"total_dead={summary['dead_count_total']} | "
                f"dead_ratio={summary['dead_ratio_mean']:.3f}"
            )
            for i in range(len(usage) - 1):  # exclude 'summary' key
                layer_key = f'layer_{i}'
                if layer_key in usage:
                    u = usage[layer_key]
                    self.logger.info(
                        set_color(f"    L{i}", "cyan") +
                        f": ppl={u['perplexity']:.1f}/{self.model.rq.n_e_list[i]} | "
                        f"dead={u['dead_count']}/{self.model.rq.n_e_list[i]} | "
                        f"usage=[{u['usage_min']:.1f}, {u['usage_mean']:.1f}, {u['usage_max']:.1f}]"
                    )
        except Exception as e:
            self.logger.debug(f"  Could not get codebook usage: {e}")

        return collision_rate

    def _save_checkpoint(self, epoch, collision_rate=1, ckpt_file=None):

        ckpt_path = os.path.join(self.ckpt_dir,ckpt_file) if ckpt_file \
            else os.path.join(self.ckpt_dir, 'epoch_%d_collision_%.4f_model.pth' % (epoch, collision_rate))
        state = {
            "args": self.args,
            "epoch": epoch,
            "best_loss": self.best_loss,
            "best_collision_rate": self.best_collision_rate,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, ckpt_path, pickle_protocol=4)

        self.logger.info(
            set_color("Saving current", "blue") + f": {ckpt_path}"
        )

        return ckpt_path

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, loss, recon_loss):
        train_loss_output = (
            set_color("epoch %d training", "green")
            + " ["
            + set_color("time", "blue")
            + ": %.2fs, "
        ) % (epoch_idx, e_time - s_time)
        train_loss_output += set_color("train loss", "blue") + ": %.4f" % loss
        train_loss_output +=", "
        train_loss_output += set_color("reconstruction loss", "blue") + ": %.4f" % recon_loss
        return train_loss_output + "]"


    def fit(self, data):

        cur_eval_step = 0

        for epoch_idx in range(self.epochs):
            # train
            training_start_time = time()
            train_loss, train_recon_loss = self._train_epoch(data, epoch_idx)
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss, train_recon_loss
            )
            self.logger.info(train_loss_output)

            # ---- Per-epoch codebook stats (always logged, even without eval) ----
            self._log_codebook_stats(epoch_idx)

            # eval
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                collision_rate = self._valid_epoch(data)

                if train_loss < self.best_loss:
                    self.best_loss = train_loss
                    self._save_checkpoint(epoch=epoch_idx, ckpt_file=self.best_loss_ckpt)

                if collision_rate < self.best_collision_rate:
                    self.best_collision_rate = collision_rate
                    cur_eval_step = 0
                    self._save_checkpoint(epoch_idx, collision_rate=collision_rate,
                                          ckpt_file=self.best_collision_ckpt)
                else:
                    cur_eval_step += 1

                # 收集结构化 metrics（包含 codebook usage）
                self._collect_metrics(epoch_idx, train_loss, train_recon_loss, collision_rate)


                valid_end_time = time()
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("collision_rate", "blue")
                    + ": %f]"
                ) % (epoch_idx, valid_end_time - valid_start_time, collision_rate)

                self.logger.info(valid_score_output)
                ckpt_path = self._save_checkpoint(epoch_idx, collision_rate=collision_rate)
                now_save = (-collision_rate, ckpt_path)
                if len(self.newest_save_queue) < self.save_limit:
                    self.newest_save_queue.append(now_save)
                    heapq.heappush(self.best_save_heap, now_save)
                else:
                    old_save = self.newest_save_queue.pop(0)
                    self.newest_save_queue.append(now_save)
                    if collision_rate < -self.best_save_heap[0][0]:
                        bad_save = heapq.heappop(self.best_save_heap)
                        heapq.heappush(self.best_save_heap, now_save)

                        if bad_save not in self.newest_save_queue:
                            delete_file(bad_save[1])

                    if old_save not in self.best_save_heap:
                        delete_file(old_save[1])



        return self.best_loss, self.best_collision_rate




