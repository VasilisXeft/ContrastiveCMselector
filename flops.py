import torch
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from build.model_builder import build_model

# 1) Build the model
model, _ = build_model("configs/config.yaml", pretrained_weights=None, freeze_encoders=True)
model.eval()

B = 1

# 2) Create individual tensors instead of a dictionary
eeg = torch.randn(B, 32, 7680)
ecg = torch.randn(B, 3, 7680)
eda = torch.randn(B, 1, 7680)
tmp = torch.randn(B, 1, 7680)
rsp = torch.randn(B, 1, 7680)
eye = torch.randn(B, 3, 1800)
signal_quality = torch.randn(B, 6)

# Notice we OMIT 'targets' here. Usually, passing targets triggers loss
# computation, which creates scalar tensors that often break the JIT tracer.

# 3) Create the Wrapper
class FLOPWrapper(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, eeg, ecg, eda, tmp, rsp, eye, sq):
        # Reconstruct the dictionary inside the forward pass
        batch = {
            "eeg": eeg,
            "ecg": ecg,
            "eda": eda,
            "tmp": tmp,
            "rsp": rsp,
            "eye": eye,
            "signal_quality": sq
        }

        # Run the model
        out = self.base_model(batch)

        # JIT Tracer HATES dictionaries.
        # If your model returns a dict, flatten it into a tuple of tensors!
        if isinstance(out, dict):
            return tuple(out.values())
        return out

# Instantiate wrapper
wrapper = FLOPWrapper(model)

# 4) Parameters (You can still run this on the base model)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("=" * 60)
print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print("=" * 60)

# 5) FLOPs Analysis
with torch.no_grad():
    # Pass the inputs as a tuple of raw tensors to the wrapper
    inputs = (eeg, ecg, eda, tmp, rsp, eye, signal_quality)

    flops = FlopCountAnalysis(wrapper, inputs)
    flops.unsupported_ops_warnings(False)

    total_macs = flops.total()
    print("=" * 60)
    print(f"Total MACs (single forward, B={B}): {total_macs:,} ({total_macs/1e6:.2f} MMac)")
    print(f"Approx. FLOPs (MACs x2):             {total_macs*2/1e6:.2f} MFLOPs")
    print("=" * 60)

    print("\nPer-module MACs breakdown:")
    for name, macs in flops.by_module().items():
        if macs > 0:
            print(f"  {name:50s} {macs:>15,} MACs")