import torch
import torch.nn.functional as F

class Distiller(torch.nn.Module):
    def __init__(self, teacher, student, Dt=128, Ds=1792, D_common=512,
                 temp=4.0, alpha=0.5, beta=5e-2, device="cpu", feat_align="label_aware"):
        super().__init__()
    
        for p in teacher.parameters(): p.requires_grad=False
        self.teacher = teacher.eval()   # frozen
        
        self.student = student.train()

        # projection heads
        self.teacher_proj = torch.nn.Linear(Dt, D_common).to(device)
        self.student_proj = torch.nn.Linear(Ds, D_common).to(device)  # EfficientNet-B3

        # FitNet regressor
        self.hint_regressor = torch.nn.Sequential(
            torch.nn.Linear(Ds, Dt),
            torch.nn.ReLU(),
            torch.nn.Linear(Dt, Dt)
        ).to(device)        
        
        self.temperature = temp
        self.alpha = alpha    
        self.beta = beta
        
        self.feat_align = feat_align
        self.to(device)
    
    def forward(self, mesh, image, label):
        
        self.teacher.eval()
        with torch.no_grad():
            #x, edge_index, batch = mesh.x, mesh.edge_index, getattr(mesh, 'batch', None)
            #t_feat, t_logits = self.teacher(x, edge_index, batch, return_features=True)
            t_feat, t_logits = self.teacher(mesh, return_features=True)
            
        s_feat, s_logits = self.student(image, return_features=True)

        # projections
        t_latent = self.teacher_proj(t_feat)
        s_latent = self.student_proj(s_feat)

        if len(s_feat.shape) == 4:
            s_feat_flat = F.adaptive_avg_pool2d(s_feat, 1).flatten(1)
        else:
            s_feat_flat = s_feat
        s_hint = self.hint_regressor(s_feat_flat)
        
        t_hint = F.normalize(t_feat.detach(), dim=1)
        s_hint = F.normalize(s_hint, dim=1)

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
        if self.feat_align == "label_aware":
            L_feat = label_aware_contrastive(s_hint, t_hint, label)
            feat_a, feat_b = t_hint, s_hint
        elif self.feat_align  == "contrastive":
            L_feat = contrastive(s_hint, t_hint)
            feat_a, feat_b = t_hint, s_hint
        elif self.feat_align  == "fitnet":
            # naive hint learning or FitNet feaure alignment  
            L_feat = fitnet_cosine(s_hint, t_hint)
            feat_a, feat_b = t_hint, s_hint
        elif self.feat_align  == "fitnet_smooth":
            # naive hint learning or FitNet feaure alignment  
            L_feat = F.smooth_l1_loss(s_hint, t_hint)
            feat_a, feat_b = t_hint, s_hint
        else:
            # instance level feature matching
            L_feat = 1 - F.cosine_similarity(s_latent, t_latent, dim=1).mean()
            feat_a, feat_b = t_latent, s_latent

        L_total = L_ce + self.alpha * L_kd + self.beta * L_feat    
        return L_total, L_ce, L_kd, L_feat, s_logits, feat_a, feat_b
            
def label_aware_contrastive(s_hint, t_hint, labels, tau=0.1):
    logits = torch.matmul(s_hint, t_hint.T) / tau

    # Create mask for positives
    labels = labels.unsqueeze(1)
    mask = (labels == labels.T).float()

    # Log-softmax
    log_prob = F.log_softmax(logits, dim=1)

    # Only keep positives
    loss = -(mask * log_prob).sum(dim=1) / mask.sum(dim=1)

    return loss.mean()

def contrastive(s_hint, t_hint, tau=0.1):
    logits_st = torch.matmul(s_hint, t_hint.T) / tau
    logits_ts = torch.matmul(t_hint, s_hint.T) / tau

    # Targets = correct pair index
    targets = torch.arange(logits_st.size(0)).to(logits_st.device)

    L_st = F.cross_entropy(logits_st, targets)
    L_ts = F.cross_entropy(logits_ts, targets)

    # Contrastive loss
    loss = (L_st + L_ts) / 2

    return loss

def fitnet_cosine(s_hint, t_hint):
    loss =  1 - F.cosine_similarity(s_hint, t_hint, dim=1).mean() 
    
    return loss