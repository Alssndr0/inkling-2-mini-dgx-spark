# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ---- recipe change notice ----
# Modified copy of vllm/models/inkling2/speculator.py from the vllm/vllm-openai:inkling-2-mini image (vLLM 7cb5dae, Apache-2.0).
# Changed: the DSpark drafter can score a pruned draft vocabulary (INKLING_DRAFT_VOCAB, INKLING_DRAFT_VOCAB_N);
# rejection sampling against the full target distribution keeps the output distribution unchanged.
# Change notice added for the inkling-2-mini-dgx-spark recipe, 2026-11 (Apache-2.0).
# ---- end of recipe change notice ----
"""DSpark speculator for the Inkling2 draft: stores the raw target features
(each draft layer mixes them itself) and runs its recurrent Markov head."""

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model


class Inkling2DSparkSpeculator(DSparkSpeculator):
    _speculator_name = "Inkling2 DSpark"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        config = self.draft_model_config.hf_config
        assert not self.sample_from_anchor
        assert self._draft_topk is None and not self.enable_adaptive_verification
        self.hidden_states = torch.zeros(
            self.max_num_tokens,
            len(config.target_layer_ids) * config.hidden_size,
            dtype=self.dtype,
            device=device,
        )
        # combine_hidden_states is the identity: store features eagerly.
        self._aux_staging = None
        # Read by prepare_context_anchor inside the FULL draft graph.
        self._num_rejected = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        self.f_rnn = torch.zeros(
            self.max_num_reqs, config.hidden_size, dtype=self.dtype, device=device
        )

    def draft_logits_spec(self, vllm_config: VllmConfig) -> tuple[torch.dtype, float]:
        # Draft base logits come from a bf16 GEMM, unlike the target's fp32 head.
        return self.dtype, 0.0

    def load_draft_model(
        self, target_model: nn.Module, target_attn_layer_names: set[str]
    ) -> nn.Module:
        model = load_dspark_model(target_model, self.vllm_config)
        model.attach_target(target_model.get_language_model())
        # Pruned draft vocabulary (campaign patch "dvocab"): as DSparkSpeculator.load_draft_model, scatter the
        # draft-vocabulary logits into the target vocabulary (-inf elsewhere) before sampling and rejection.
        if self.use_draft_logits and model.draft_id_to_target_id is not None:
            d2t = model.draft_id_to_target_id
            self._d2t_scatter_index = torch.arange(d2t.shape[0], device=d2t.device) + d2t
            self._draft_scatter_buf = torch.full(
                (self.max_num_reqs, self.vocab_size),
                float("-inf"),
                dtype=self.draft_logits_spec(self.vllm_config)[0],
                device=self.device,
            )
        return model

    def prepare_context_anchor(
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        num_rejected: torch.Tensor | None,
    ) -> None:
        assert num_rejected is not None
        ends = query_start_loc[1 : num_reqs + 1] - num_rejected[:num_reqs]
        rows = self.hidden_states[(ends - 1).clamp_min(0).long()]
        self.f_rnn[:num_reqs] = self.model.compute_f_rnn(rows)

    def _sample_sequential(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec
        hidden = head_hidden[self.sample_indices[:num_sample]]
        base_logits = self.model.compute_base_logits(hidden).view(num_reqs, n_spec, -1)
        context = self.model.model.markov_head.context(hidden, self.f_rnn[:num_reqs])
        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
        state = hidden.new_zeros(num_reqs, self.model.config.markov_rank)
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]
        for i in range(n_spec):
            state, bias = self.model.markov_step(state, prev, context[i])
            if self.draft_logits is None:
                prev = self.model.markov_argmax(base_logits[:, i], bias)
            else:
                logits = self.model.markov_logits(base_logits[:, i], bias)
                prev = self._sample_logits(logits, idx_map[:, i], sample_pos[:, i], i)
            self.draft_tokens[:num_reqs, i] = prev
