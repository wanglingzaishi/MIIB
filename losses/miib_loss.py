import torch.nn.functional as func
import torch
import torch.nn as nn
from utils import all_gather_batch_with_grad
from losses.cos_similarity import Cosine_loss
from losses.entropy import entropy_gradeint


class MIIBLoss(nn.Module):
    """MIIB modality-specific, fusion, and synergy objective.

    Uses alignment, entropy-gradient proxies, and a cosine hinge constraint.
    Original multimodal infrastructure: Dufumier et al., CoMM, ICLR 2025.

    The objective decomposes into five terms:

    ``alignment``
        Similarity between the fused prototype and the unimodal / cross-view
        representations. Maximised.
    ``uniformity``
        vMF entropy-gradient of the fused and unimodal representations.
        Maximised (prevents collapse).
    ``synergy_alignment``
        Similarity between the synergy representation and the fused
        representation of the other view. Maximised.
    ``synergy_uniformity``
        vMF entropy-gradient of the synergy representation. Maximised.
    ``synergy_diversity``
        Cosine hinge pushing the synergy representation away from the
        unimodal streams, i.e. it must carry information that no single
        modality carries. Minimised.

    All sign handling is centralised in :meth:`forward`: the four first terms
    are *maximised* and therefore enter the total loss with a negative sign,
    while ``synergy_diversity`` is *minimised* and enters with a positive
    sign.
    """

    def __init__(self,
                 temperature=0.1,
                 synergy_uniformity_weight=0.5,
                 fusion_uniformity_weight=1.0,
                 unimodal_uniformity_weight=1.0,
                 diversity_margin=-0.5,
                 synergy_weight=1.0,
                 weights=None):
        super().__init__()

        self.cosine_loss = Cosine_loss(diversity_margin)
        self.temperature = temperature
        self.weights = weights
        self.fusion_uniformity_weight = fusion_uniformity_weight
        self.unimodal_uniformity_weight = unimodal_uniformity_weight
        self.synergy_uniformity_weight = synergy_uniformity_weight
        self.synergy_weight = synergy_weight

    def compute_loss_terms(self, z, fusion_z1, fusion_z2, z11, z22):
        """Compute the five MIIB objective terms.

        :param z: list of normalised embeddings ``[prototype_v1, prototype_v2,
            unimodal_v1..., unimodal_v2...]``.
        :param fusion_z1: fused representation of view 1.
        :param fusion_z2: fused representation of view 2.
        :param z11: synergy representation of view 1.
        :param z22: synergy representation of view 2.
        :return: ``(alignment, uniformity, synergy_alignment,
            synergy_uniformity, synergy_diversity, retrieval_acc)``.
        """
        N = len(fusion_z1)  # (batchsize)

        alignment = 0
        uniformity = 0
        retrieval_acc = 0
        n = int((len(z) - 2) / 2)
        # 两个增强的单模态
        z1 = z[2:n + 2]
        z2 = z[2 + n:]

        # L_mf: alignment between the fused prototype and the cross-view
        # unimodal representations.
        similarity_logits_total1 = torch.einsum('nc,nc->n', [z[0], fusion_z2])
        similarity_logits_total2 = torch.einsum('nc,nc->n', [z[1], fusion_z1])
        with torch.no_grad():
            sim_zij_s = (z[0] @ fusion_z2.T) / self.temperature
            pred = torch.argmax(sim_zij_s, dim=1)
            correct = pred.eq(torch.arange(N, device=sim_zij_s.device)).sum()
            retrieval_acc += 100 * correct / N
        alignment += similarity_logits_total1.mean()
        alignment += similarity_logits_total2.mean()
        uniformity += self.fusion_uniformity_weight * entropy_gradeint(z[0])
        uniformity += self.fusion_uniformity_weight * entropy_gradeint(z[1])

        # L_MS: alignment between the unimodal representations and the
        # cross-view fused representations.
        for i in range(n):
            similarity_logits_1 = torch.einsum('nc,nc->n', [z1[i], fusion_z2])
            similarity_logits_2 = torch.einsum('nc,nc->n', [z2[i], fusion_z1])

            alignment += similarity_logits_1.mean()
            alignment += similarity_logits_2.mean()

            uniformity += self.unimodal_uniformity_weight * entropy_gradeint(z1[i])
            uniformity += self.unimodal_uniformity_weight * entropy_gradeint(z2[i])

        # L_s: synergy branch.
        synergy_alignment = 0
        synergy_uniformity = 0
        synergy_diversity = 0

        similarity1_S = torch.einsum("nc,nc->n", [z11, fusion_z2])
        synergy_alignment += similarity1_S.mean()
        similarity2_S = torch.einsum("nc,nc->n", [z22, fusion_z1])
        synergy_alignment += similarity2_S.mean()

        synergy_uniformity += self.synergy_uniformity_weight * entropy_gradeint(z11)
        synergy_uniformity += self.synergy_uniformity_weight * entropy_gradeint(z22)

        for i in range(n):
            # 之前 transformer 的单模态的两个增强数据相加
            synergy_diversity += self.cosine_loss(z11, z1[i])
            synergy_diversity += self.cosine_loss(z22, z2[i])

        return (alignment / len(z),
                uniformity / len(z),
                self.synergy_weight * synergy_alignment / 2,
                self.synergy_weight * synergy_uniformity / 2,
                synergy_diversity / n,
                retrieval_acc)

    def forward(self, outputs):
        """
        :param outputs: Dict
            Dictionary with keys:
                - "aug1_embed", List of tensors with shape (bsize, feature_dim), 1st aug.
                - "aug2_embed", List of tensors with shape (bsize, feature_dim), 2nd aug.
                - "prototype", integer indicating where the multimodal representation Z
                    is stored in "aug1_embed" and "aug2_embed".
        :return: {"loss": torch.Tensor(float), "ssl_acc": torch.Tensor(float)}
        """
        # Prepare embeddings (normalize + gather across all GPU)
        z1, z11, z2, z22, prototype = outputs["aug1_fus1"], outputs["aug1_fus2"], outputs["aug2_fus1"], outputs["aug2_fus2"], outputs["prototype"]
        assert len(z1) == len(z2)
        n_emb = len(z1)
        z1 = [func.normalize(z, p=2, dim=-1) for z in z1]
        z2 = [func.normalize(z, p=2, dim=-1) for z in z2]
        z11 = [func.normalize(z, p=2, dim=-1) for z in z11]
        z22 = [func.normalize(z, p=2, dim=-1) for z in z22]
        Z = all_gather_batch_with_grad(z1 + z2)
        Z1 = all_gather_batch_with_grad(z11 + z22)
        z1, z2 = Z[:n_emb], Z[n_emb:]
        z11, z22 = Z1[:n_emb], Z1[n_emb:]
        z = [z1[prototype], z2[prototype]] + z1[:-1] + z2[:-1]

        (alignment, uniformity, synergy_alignment, synergy_uniformity,
         synergy_diversity, retrieval_acc) = self.compute_loss_terms(
            z, z1[prototype], z2[prototype], z11[-1], z22[-1])

        # Sign convention: the first four terms are maximised (negative sign),
        # the diversity hinge is minimised (positive sign).
        maximised = [alignment, uniformity, synergy_alignment, synergy_uniformity]
        if self.weights is not None:
            weights = torch.tensor(self.weights, device=z1[0].device)
            if len(weights) != 5:
                raise ValueError(
                    f"'weights' must hold 5 entries (alignment, uniformity, "
                    f"synergy_alignment, synergy_uniformity, synergy_diversity), "
                    f"got {len(weights)}.")
            loss = (-torch.sum(torch.stack(maximised) * weights[:4])
                    + weights[4] * synergy_diversity)
        else:
            loss = -sum(maximised) + synergy_diversity

        return {"loss": loss, "ssl_acc": retrieval_acc,
                "alignment": alignment, "uniformity": uniformity,
                "synergy_alignment": synergy_alignment,
                "synergy_uniformity": synergy_uniformity,
                "synergy_diversity": synergy_diversity}

    def __str__(self):
        return "{}(temp={})".format(type(self).__name__, self.temperature)
