import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import threading
import time
import os
from collections import deque
from utils import CharTokenizer, get_batch

# ------------------ 模型组件 ------------------
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        out = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * out + self.beta

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        y = self.proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        pos_emb = self.position_embedding(pos)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        self.train()
        return idx

# ------------------ 训练配置 ------------------
class Config:
    def __init__(self):
        # 模型结构
        self.block_size = 256
        self.vocab_size = None
        self.n_embd = 512
        self.n_head = 8
        self.n_layer = 8
        self.dropout = 0.1
        # 训练超参
        self.batch_size = 32
        self.gradient_accumulation_steps = 2   # 实际 batch = batch_size * grad_acc
        self.learning_rate = 3e-4
        self.max_iters = 5000
        self.warmup_iters = 500
        self.weight_decay = 0.01
        self.betas = (0.9, 0.95)
        self.grad_norm_clip = 1.0
        self.scheduler = 'cosine'   # 'cosine' or 'none'
        # 评估与生成
        self.eval_interval = 200
        self.eval_iters = 5
        self.early_stop_patience = 5   # 验证损失不再下降时停止
        # 生成默认值
        self.temperature = 0.8
        self.top_k = 40
        self.top_p = 0.9
        # 系统
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mixed_precision = False   # 如需加速可开启，需要 torch.cuda.amp
        self.cpu_threads = 0   # 0=使用全部CPU核心，可限制以降低占用

# ------------------ 训练管理器 ------------------
class ModelManager:
    def __init__(self):
        self.config = Config()
        self.model = None
        self.tokenizer = None
        self.train_data = None
        self.val_data = None
        self.is_training = False
        self.training_thread = None
        self.callbacks = {}
        self.current_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self._last_train_loss = None
        self._training_error = None
        # 消息历史缓冲区，用于 SSE 重连或晚连接时回放
        self._message_history = deque(maxlen=50)
        self._history_lock = threading.Lock()

    def load_dataset(self, file_path):
        self._dataset_path = file_path
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        self.tokenizer = CharTokenizer(text)
        self.config.vocab_size = self.tokenizer.vocab_size
        data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        n = int(0.9 * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]
        return {
            'vocab_size': self.tokenizer.vocab_size,
            'total_chars': len(text),
            'sample': text[:300]
        }

    def create_model(self):
        self.model = GPT(self.config)
        self.model.to(self.config.device)
        return sum(p.numel() for p in self.model.parameters())

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config.__dict__,
            'tokenizer_stoi': self.tokenizer.stoi,
            'tokenizer_itos': self.tokenizer.itos,
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.config.device)
        # 恢复 tokenizer
        self.tokenizer = CharTokenizer("")  # dummy
        self.tokenizer.stoi = checkpoint['tokenizer_stoi']
        self.tokenizer.itos = checkpoint['tokenizer_itos']
        self.tokenizer.vocab_size = len(self.tokenizer.stoi)
        self.config.vocab_size = self.tokenizer.vocab_size
        # 恢复 config（仅恢复必要的结构参数）
        saved_config = checkpoint['config']
        for k in ['block_size', 'n_embd', 'n_head', 'n_layer', 'dropout', 'vocab_size']:
            if k in saved_config:
                setattr(self.config, k, saved_config[k])
        self.create_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.config.device)

    def set_callback(self, name, func):
        self.callbacks[name] = func

    def _emit(self, name, data):
        # 自动添加 type 字段，确保前端 SSE 能正确路由
        if isinstance(data, dict):
            data = dict(data)  # 复制避免修改原始对象
            if 'type' not in data:
                data['type'] = name
        # 存入历史缓冲区
        with self._history_lock:
            self._message_history.append(data)
        if name in self.callbacks:
            try:
                self.callbacks[name](data)
            except Exception as e:
                print(f"[Callback Error] {name}: {e}")

    def get_message_history(self):
        """获取当前消息历史列表（用于 SSE 连接时回放）"""
        with self._history_lock:
            return list(self._message_history)

    def train(self, hyperparams):
        if self.is_training:
            return
        # 更新超参数
        for key, value in hyperparams.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        # 确保 gradient_accumulation_steps >= 1
        if self.config.gradient_accumulation_steps < 1:
            self.config.gradient_accumulation_steps = 1
        # 应用 CPU 线程限制
        if self.config.device == 'cpu' and self.config.cpu_threads > 0:
            torch.set_num_threads(self.config.cpu_threads)
        self.create_model()
        self.is_training = True
        self.current_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_thread = threading.Thread(target=self._training_worker)
        self.training_thread.daemon = True
        self.training_thread.start()

    def _training_worker(self):
        # 优化器
        optimizer = AdamW(self.model.parameters(),
                          lr=self.config.learning_rate,
                          betas=self.config.betas,
                          weight_decay=self.config.weight_decay)
        # 学习率调度
        if self.config.scheduler == 'cosine':
            scheduler1 = LinearLR(optimizer, start_factor=1e-5, end_factor=1.0, total_iters=self.config.warmup_iters)
            scheduler2 = CosineAnnealingLR(optimizer, T_max=self.config.max_iters - self.config.warmup_iters)
            scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[self.config.warmup_iters])
        else:
            scheduler = None

        # 混合精度（兼容新旧版本torch）
        try:
            from torch.amp import GradScaler
            scaler = GradScaler('cuda', enabled=self.config.mixed_precision and self.config.device == 'cuda')
        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled=self.config.mixed_precision and self.config.device == 'cuda')

        # 计时器
        import time
        step_start_time = time.time()
        tokens_processed = 0

        self._emit('status', {'type': 'start', 'message': f'训练开始，设备: {self.config.device}'})

        for step in range(self.config.max_iters):
            if not self.is_training:
                break
            self.current_step = step

            # 梯度累积内循环
            accumulated_loss = 0.0
            for micro_step in range(self.config.gradient_accumulation_steps):
                xb, yb = get_batch(self.train_data, self.config.block_size,
                                   self.config.batch_size, self.config.device)
                tokens_processed += xb.numel()
                try:
                    from torch.amp import autocast
                    with autocast('cuda', enabled=self.config.mixed_precision and self.config.device == 'cuda'):
                        _, loss = self.model(xb, yb)
                except Exception:
                    with torch.cuda.amp.autocast(enabled=self.config.mixed_precision and self.config.device == 'cuda'):
                        _, loss = self.model(xb, yb)
                loss = loss / self.config.gradient_accumulation_steps
                accumulated_loss += loss.item()
                scaler.scale(loss).backward()

            # 梯度裁剪
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler:
                scheduler.step()

            self._last_train_loss = accumulated_loss

            # 定期评估
            if step % self.config.eval_interval == 0 or step == self.config.max_iters - 1:
                step_elapsed = time.time() - step_start_time
                tokens_per_sec = tokens_processed / max(step_elapsed, 1e-6)
                step_start_time = time.time()
                tokens_processed = 0

                val_loss = self._estimate_loss()
                ppl = math.exp(val_loss)
                lr = optimizer.param_groups[0]['lr']
                # 生成样例
                sample = self._generate_sample(prompt="ROMEO:", max_new=150)
                self._emit('progress', {
                    'step': step,
                    'train_loss': accumulated_loss,
                    'val_loss': val_loss,
                    'ppl': ppl,
                    'lr': lr,
                    'grad_norm': float(grad_norm),
                    'tokens_per_sec': round(tokens_per_sec, 1),
                    'sample': sample
                })
                # 早停逻辑
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.save_model('models/best_model.pt')
                    self._emit('status', {'type': 'save', 'message': f'新最佳模型 (step {step}) 已保存'})
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.config.early_stop_patience:
                        self._emit('status', {'type': 'early_stop', 'message': f'早停于 step {step}'})
                        break

        self.is_training = False
        self._emit('status', {'type': 'end', 'message': '训练结束'})

    def _estimate_loss(self):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for _ in range(self.config.eval_iters):
                xv, yv = get_batch(self.val_data, self.config.block_size,
                                   self.config.batch_size, self.config.device)
                _, loss = self.model(xv, yv)
                losses.append(loss.item())
        self.model.train()
        return sum(losses) / len(losses)

    def _generate_sample(self, prompt, max_new=100):
        input_ids = self.tokenizer.encode(prompt)
        if len(input_ids) > self.config.block_size - 1:
            input_ids = input_ids[-(self.config.block_size - 1):]
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=self.config.device).unsqueeze(0)
        output_ids = self.model.generate(input_tensor, max_new_tokens=max_new,
                                         temperature=self.config.temperature,
                                         top_k=self.config.top_k,
                                         top_p=self.config.top_p)
        return self.tokenizer.decode(output_ids[0].tolist())

    def generate_response(self, prompt, max_new=200, temperature=None, top_k=None, top_p=None):
        if self.model is None:
            return "模型未加载，请先训练或加载已有模型。"
        # 使用当前配置或临时覆盖
        temp = temperature if temperature is not None else self.config.temperature
        tk = top_k if top_k is not None else self.config.top_k
        tp = top_p if top_p is not None else self.config.top_p
        # 截断 prompt 到 block_size - 1
        max_len = self.config.block_size - 1
        if len(prompt) > max_len:
            prompt = prompt[-max_len:]
        input_ids = self.tokenizer.encode(prompt)
        if len(input_ids) == 0:
            return "[输入包含未知字符，无法生成]"
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=self.config.device).unsqueeze(0)
        output_ids = self.model.generate(input_tensor, max_new_tokens=max_new,
                                         temperature=temp, top_k=tk, top_p=tp)
        full_text = self.tokenizer.decode(output_ids[0].tolist())
        # 去掉原始 prompt 部分
        if full_text.startswith(prompt):
            response = full_text[len(prompt):]
        else:
            response = full_text
        # 记录推理日志
        self._emit('inference_log', {
            'prompt_preview': prompt[-80:],
            'response': response.strip()[:300],
            'params': {'temperature': temp, 'top_k': tk, 'top_p': tp, 'max_new': max_new},
            'input_tokens': len(input_ids),
            'output_tokens': len(output_ids[0].tolist()) - len(input_ids)
        })
        return response.strip()

    def stop_training(self):
        self.is_training = False