from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmpose.core.distributed_wrapper import DistributedDataParallelWrapper

from mmcv.runner import HOOKS, Hook

import torch
import torch.nn.functional as F

@HOOKS.register_module()
class CPSInitHook(Hook):
    """Initialize Clustered Prototypes Superivison at certain epoch. (use for ViTPose + Proto only)

    Args:
        init_epoch (int): Epoch to initialize.
        cluster_options (list): options to test for the number of clusters
        cluster_iter (int): iterations for clustering.
        beta (int): scaling coefficient for k means
    """

    def __init__(self,
                 init_epoch=-1,
                 cluster_options=[],
                 cluster_iter=-1,
                 beta=20,
                 **kwargs):
        self.init_epoch = init_epoch
        self.cluster_options = cluster_options
        self.cluster_iter = cluster_iter
        self.beta = beta

    def get_clusters(self, proto, n_cluster, iters):
        J, M, D = proto.shape
        proto_flat = proto.reshape(J*M, D)
        
        # initialize centroids
        initial_sample_idxs = [i*M for i in range(n_cluster)]
        centers = proto_flat[initial_sample_idxs]

        xs = proto_flat.unsqueeze(1) # [J*M, 1, D]
        cs = centers.unsqueeze(0) # [1, K, D]

        # run clustering
        for it in range(iters):
            sim = F.cosine_similarity(xs,cs, dim=2) # [J*M, K]
            dist = (1 - sim) * 0.5
            assign = F.softmax(-self.beta * dist, dim=1)
            cs = (assign.unsqueeze(2) * xs).sum(dim=0,keepdim=True) / assign.sum(dim=0,keepdim=True).unsqueeze(2) # [1, K, D]
        
        # get cluster assignments
        sim = F.cosine_similarity(xs,cs, dim=2) # [J*M, K]
        dist = (1 - sim) * 0.5
        assign = F.softmax(-self.beta * dist, dim=1) # [J*M, K]

        #cs = cs.squeeze(0)
        # get top-M closest prototypes with duplication
        idxs = torch.argsort(dist, dim=1)
        idxs_topm = idxs[:M, :].T

        proto_cst = torch.zeros((n_cluster, M, D), dtype=proto.dtype, device=proto.device)
        for ci in range(n_cluster):
            ci_idxs = idxs_topm[ci, :]
            proto_cst[ci] = proto_flat[ci_idxs]
        
        # get the closest cumulative keypoint indices for each clustered prototypes
        closest_kpt_idxs = idxs_topm[:,0].div(M).floor().long()
        
        # get reconstruction error
        amat = cs.squeeze(0).T
        bmat = xs.squeeze(1).T
        res = torch.linalg.lstsq(amat, bmat)
        w = res.solution
        b_hat = amat@w
        err = torch.sum((bmat - b_hat) ** 2)

        return assign, closest_kpt_idxs, proto_cst, err


    def before_train_epoch(self, runner):
        epoch = runner.epoch
        if epoch != self.init_epoch:
            return

        model = runner.model
        if isinstance(model, (MMDataParallel, MMDistributedDataParallel, DistributedDataParallelWrapper)):
            model = model.module
        proto = model.proto_head.kpt_prototype.prototypes.detach().type(torch.float32)
        proto = F.normalize(proto, p=2, dim=-1)
        n_proto = proto.shape[0]
        
        best_n_cluster = -1
        best_error = float("Inf")
        best_weighted_error = float("Inf")
        best_assigns = None
        best_closest_kpt_idxs = None
        best_centroids = None
        for n_cluster in self.cluster_options:
            if n_cluster >= n_proto:
                continue
            assigns, closest_kpt_idxs, centroids, error = self.get_clusters(proto, n_cluster, self.cluster_iter)
            weighted_error = error * (n_cluster / n_proto)**2
            if weighted_error < best_weighted_error:
                best_n_cluster = n_cluster
                best_error = error
                best_weighted_error = weighted_error
                best_assigns = assigns
                best_closest_kpt_idxs = closest_kpt_idxs
                best_centroids = centroids
            #print(f"{n_cluster} clusters error: {error:.4e}, weighted error: {weighted_error:.4e}")
        #print(f"CPSInitHook result. best n_cluster: {best_n_cluster}, error: {best_error:.4e}, weighted error: {weighted_error:.4e}")

        #print(f"proto.shape: {proto.shape}, n_cluster: {n_cluster}")
        #print(f"best_assigns.shape: {best_assigns.shape}, best_closest_kpt_idxs.shape: {best_closest_kpt_idxs.shape}, best_centroids.shape: {best_centroids.shape}")

        # save results into the cluster_prototypes
        model.proto_head.set_cluster_prototypes(best_assigns, best_closest_kpt_idxs, best_centroids)
        