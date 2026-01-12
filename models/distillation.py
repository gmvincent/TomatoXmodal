import torch
import torch.nn.functional as F

class Distiller(torch.nn.Module):
    def __init__(self, teacher, student, Dt=128, Ds=1536, D_common=512,
                 temp=6.0, alpha=0.5, beta=5e-2, device="cpu"):
        super().__init__()
    
        for p in teacher.parameters(): p.requires_grad=False
        self.teacher = teacher.eval()   # frozen
        
        self.student = student.train()

        # projection heads
        self.teacher_proj = torch.nn.Linear(Dt, D_common).to(device)
        self.student_proj = torch.nn.Linear(Ds, D_common).to(device)  # EfficientNet-B3

        self.temperature = temp
        self.alpha = alpha    
        self.beta = beta
        
        self.to(device)

    def forward(self, mesh, image, label):
        
        self.teacher.eval()
        with torch.no_grad():
            x, edge_index, batch = mesh.x, mesh.edge_index, getattr(mesh, 'batch', None)
            t_feat, t_logits = self.teacher(x, edge_index, batch, return_features=True)

        s_feat, s_logits = self.student(image, return_features=True)

        # projection
        t_latent = self.teacher_proj(t_feat)
        s_latent = self.student_proj(s_feat)

        # ----- losses -----

        # CE loss
        L_ce = F.cross_entropy(s_logits, label)

        # Logit KD loss
        T = self.temperature
        L_kd = F.kl_div(
            F.log_softmax(s_logits / T, dim=1),
            F.softmax(t_logits / T, dim=1),
            reduction="batchmean"
        ) * (T*T)

        # Feature alignment
        L_feat = 1 - F.cosine_similarity(s_latent, t_latent, dim=1).mean()

        L_total = L_ce + self.alpha * L_kd + self.beta * L_feat
        
        return L_total, L_ce, L_kd, L_feat, s_logits
