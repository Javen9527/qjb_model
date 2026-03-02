"""
Transformer 模型 (Attention is All You Need, Vaswani et al., 2017).
基于 nn.Transformer 的编码器-解码器结构，用于序列到序列任务。
"""

import logging
import math
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

log = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """ sinusoidal 位置编码 """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerModel(LightningModule):
    """
    Transformer 编码器-解码器模型。
    输入: src (B, src_len), tgt (B, tgt_len)，整型 token id
    输出: logits (B, tgt_len, vocab_size)，用于下一 token 预测
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        padding_idx: int = 0,
        max_len: int = 128,
        lr: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        log.info(
            "TransformerModel 初始化: vocab_size=%s, d_model=%s, nhead=%s, layers=%s/%s",
            vocab_size,
            d_model,
            nhead,
            num_encoder_layers,
            num_decoder_layers,
        )

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.hparams.d_model))
        tgt_emb = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.hparams.d_model))

        if tgt_mask is None:
            tgt_len = tgt.size(1)
            tgt_mask = self._generate_square_subsequent_mask(tgt_len, tgt.device)

        out = self.transformer(
            src_emb,
            tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return self.fc_out(out)

    def _get_loss(self, batch):
        src, tgt = batch
        tgt_input = tgt[:, :-1]
        tgt_target = tgt[:, 1:].reshape(-1)
        logits = self(src, tgt_input)
        logits_flat = logits.reshape(-1, self.hparams.vocab_size)
        loss = nn.functional.cross_entropy(
            logits_flat,
            tgt_target,
            ignore_index=self.hparams.padding_idx,
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
        return {"test_loss": loss}

    def predict_step(self, batch, batch_idx):
        src = batch[0] if isinstance(batch, (list, tuple)) else batch
        if isinstance(batch, (list, tuple)) and len(batch) > 1:
            tgt = batch[1]
        else:
            tgt = src[:, :1]
        logits = self(src, tgt)
        pred = logits.argmax(dim=-1)
        log.info("predict batch_idx=%d src_shape=%s pred_shape=%s", batch_idx, src.shape, pred.shape)
        return pred
