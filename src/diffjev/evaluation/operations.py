"""Executed tensor-shape FLOPs, separate from timed inference.

Counts GEMM/GQA/SDPA from their actual operand shapes (two FLOPs per MAC),
plus common scalar arithmetic. Unclassified operators are exported, never
silently described as a complete hardware FLOP measurement.
"""
from collections import Counter
import math
import torch
from torch.utils._python_dispatch import TorchDispatchMode


class OperationCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.flops = Counter()
        self.unclassified = Counter()
        self.shapes = Counter()
        self.sdpa_dense_bound_flops = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        name = str(func)
        count = None
        if name.startswith("aten.addmm."):
            a,b = args[1:3]
            count = 2*a.shape[-2]*a.shape[-1]*b.shape[-1] + out.numel()
        elif name.startswith(("aten.mm.", "aten.bmm.")):
            a,b = args[:2]
            count = 2*math.prod(a.shape[:-1])*a.shape[-1]*b.shape[-1]
        elif name.startswith(("aten.mv.", "aten.dot.")):
            count = 2*args[0].numel()
        elif "scaled_dot_product" in name and "backward" not in name:
            q,k,v = args[:3]
            arguments = {a.name: value for a,value in zip(func._schema.arguments,args)}
            arguments.update(kwargs or {})
            batch_heads = math.prod(q.shape[:-2])
            nq,nk = q.shape[-2],k.shape[-2]
            pairs = nq*nk
            per_pair = 2*(q.shape[-1]+v.shape[-1])+5
            self.sdpa_dense_bound_flops += batch_heads*pairs*per_pair
            # Fused causal kernels skip the upper triangle. Count its logical
            # arithmetic separately from a dense padded-tile upper bound. The
            # unfused math path actually evaluates dense masked products.
            fused = any(kind in name for kind in ("flash","efficient","cudnn"))
            if fused and arguments.get("is_causal",False):
                short=min(nq,nk)
                pairs=short*(short+1)//2+max(0,nq-nk)*nk
            count = batch_heads*pairs*per_pair
        elif name.startswith(("aten.add.","aten.sub.","aten.mul.","aten.div.","aten.neg.",
                              "aten.exp.","aten.log.","aten.pow.","aten.sqrt.","aten.rsqrt.")):
            if isinstance(out, torch.Tensor):
                count = out.numel()
        elif "softmax" in name and isinstance(out, torch.Tensor):
            count = 5*out.numel()
        elif name.startswith("aten.silu."):
            count = 4*out.numel()
        elif name.startswith(("aten.sum.","aten.mean.")):
            count = args[0].numel()
        elif "layer_norm" in name or "rms_norm" in name:
            count = 8*args[0].numel()
        if count is not None:
            self.flops[name] += int(count)
            self.shapes[(name, str([list(a.shape) for a in args if isinstance(a,torch.Tensor)]))] += 1
        else:
            self.unclassified[name] += 1
        return out

    def report(self):
        return {"gflops": sum(self.flops.values())/1e9, "by_operator": dict(self.flops),
                "sdpa_dense_bound_gflops":self.sdpa_dense_bound_flops/1e9,
                "unclassified_operator_calls": dict(self.unclassified),
                "shapes": [{"operator": n, "operand_shapes": s, "calls": c} for (n,s),c in self.shapes.items()],
                "convention": "2 FLOPs/MAC; scalar functions count as one; fused causal SDPA counts triangular logical pairs; other SDPA uses dense shapes; padded-tile hardware work may differ",
                "coverage": "GEMM and SDPA explicitly counted; audit unclassified calls before publishing a total"}


@torch.inference_mode()
def profile_request(call, cuda=False):
    """Separate operator audit and CUDA kernel-only profile, outside wall timing."""
    with OperationCounter() as counter:
        result = call()
    report = counter.report()
    report["failed"] = bool(result.get("failed"))
    report["gpu_kernel_ms"] = None
    if cuda and not report["failed"]:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
            call()
        events = prof.key_averages()
        elapsed = sum(getattr(e,"self_device_time_total",getattr(e,"self_cuda_time_total",0)) for e in events)
        report["gpu_kernel_ms"] = elapsed/1000 if elapsed > 0 else None
        report["gpu_profile_available"] = elapsed > 0
        report["profiler_operator_names"] = [e.key for e in events]
    return report
