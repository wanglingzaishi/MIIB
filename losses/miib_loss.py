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
    """

    def __init__(self, temperature=0.1,penalty = 0.5,lambda_fusion_penalty=1,lambda_single_penalty=1,margin=-0.5,penalty_fs=0.5,penalty_l=1.0,weights=None):
        super().__init__()
        self.cosine_loss=Cosine_loss(margin)
        self.temperature = temperature
        self.weights = weights
        self.penalty_fs = penalty_fs
        self.lambda_fusion_penalty = lambda_fusion_penalty
        self.lambda_single_penalty = lambda_single_penalty
        self.S_penalty = penalty
        self.penalty_l = penalty_l
        self.INF = 1e8
        self.small = 1e-8
        
    def loss_mveb(self, z, fusion_z1, fusion_z2,z11,z22):
        N = len(fusion_z1)##(batchsize)
        
        loss_aliment = 0
        loss_uniform = 0
        acc = 0
        n = int((len(z) - 2) / 2)
        # print("-------------")
        # print(n)
        #获得两个增强的单模态
        z1 = z[2:n + 2]
        z2 = z[2 + n:]
        #L_mf
        similarity_logits_total1 = torch.einsum('nc,nc->n', [z[0], fusion_z2])
        similarity_logits_total2 = torch.einsum('nc,nc->n', [z[1], fusion_z1])
        with torch.no_grad():
            sim_zij_s = (z[0] @ fusion_z2.T) / self.temperature
            pred = torch.argmax(sim_zij_s, dim=1)
            correct = pred.eq(torch.arange(N, device=sim_zij_s.device)).sum()
            acc += 100 * correct / N
        loss_aliment += similarity_logits_total1.mean()
        loss_aliment += similarity_logits_total2.mean()       
        loss_uniform += self.lambda_fusion_penalty * entropy_gradeint(z[0])
        loss_uniform += self.lambda_fusion_penalty * entropy_gradeint(z[1])
        #L_MS
        for i in range(n):
            # 计算两个相似度向量
            similarity_logits_1 = torch.einsum('nc,nc->n', [z1[i], fusion_z2])
            similarity_logits_2 = torch.einsum('nc,nc->n', [z2[i], fusion_z1])

            loss_aliment += similarity_logits_1.mean()
            loss_aliment += similarity_logits_2.mean()
            
            loss_uniform += self.lambda_single_penalty * entropy_gradeint(z1[i])
            loss_uniform += self.lambda_single_penalty *  entropy_gradeint(z2[i]) 
        
        #L_s
        aliment_S = 0
        uniform_S = 0
        no_aliment_S = 0

        similarity1_S = torch.einsum("nc,nc->n", [z11, fusion_z2])
        aliment_S += similarity1_S.mean()
        similarity2_S = torch.einsum("nc,nc->n", [z22, fusion_z1])
        aliment_S += similarity2_S.mean()
        
        uniform_S += self.S_penalty *  entropy_gradeint(z11)
        uniform_S += self.S_penalty *  entropy_gradeint(z22)
        for i in range(n):
            #之前transformer的单模态的两个增强数据相加
            no_aliment_S += self.cosine_loss(z11,z1[i])
            no_aliment_S += self.cosine_loss(z22, z2[i])   
        return loss_aliment / (len(z)), loss_uniform / len(z), acc, self.penalty_l*aliment_S /2, self.penalty_l * uniform_S / 2, -(no_aliment_S /n)
    
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
        z11 = [func.normalize(z,p=2,dim=-1) for z in z11]
        z22 = [func.normalize(z,p=2,dim=-1) for z in z22]
        Z = all_gather_batch_with_grad(z1 + z2)
        Z1= all_gather_batch_with_grad(z11+z22)
        z1, z2 = Z[:n_emb], Z[n_emb:]
        z11, z22 = Z1[:n_emb], Z1[n_emb:]
        z = [z1[prototype], z2[prototype]] + z1[:-1] + z2[:-1]

        loss = []
        acc = []
        aliment, uniform, acc, aliment_S, uniform_S, no_aliment_S = self.loss_mveb(z, z1[prototype], z2[prototype],z11[-1],z22[-1])
        if self.weights is not None:
            losses = [aliment, uniform, aliment_S, uniform_S, no_aliment_S]
            weights = torch.tensor(self.weights, device=z1[0].device)
            loss = -torch.sum(torch.stack(losses) * weights)
        else:
            loss = -aliment - uniform -  aliment_S-  uniform_S - no_aliment_S
               
        return {"loss": loss, "ssl_acc": acc, "aliment": aliment, "uniform": uniform, 
                "aliment_S": aliment_S, "uniform_S": uniform_S, "no_aliment_S": no_aliment_S}

    def __str__(self):
        return "{}(temp={})".format(type(self).__name__, self.temperature)