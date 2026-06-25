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

# --------------------------------------------------------------------
# 3) Create the Wrapper (Ο ΑΠΟΛΥΤΟΣ JIT-SAFE WRAPPER)
# --------------------------------------------------------------------
class FLOPWrapper(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, eeg, ecg, eda, tmp, rsp, eye, sq):
        # Ανακατασκευή του dictionary
        batch = {
            "eeg": eeg,
            "ecg": ecg,
            "eda": eda,
            "tmp": tmp,
            "rsp": rsp,
            "eye": eye,
            "signal_quality": sq
        }

        # Τρέχουμε το μοντέλο
        out = self.base_model(batch)

        # --- ΑΣΦΑΛΗΣ ΕΞΑΓΩΓΗ ΓΙΑ ΤΟΝ TRACER ---
        # Μαζεύουμε ΜΟΝΟ τα float tensors (αγνοώντας LongTensors, indices, κλπ)
        float_outputs = []

        def extract_floats(x):
            if isinstance(x, torch.Tensor) and x.is_floating_point():
                float_outputs.append(x)
            elif isinstance(x, dict):
                for v in x.values():
                    extract_floats(v)
            elif isinstance(x, (list, tuple)):
                for v in x:
                    extract_floats(v)

        extract_floats(out)

        # Επιστρέφουμε ένα απλό άθροισμα όλων των float εξόδων.
        # Αυτό ικανοποιεί το JIT tracer γιατί βλέπει ένα float tensor
        # που εξαρτάται άμεσα από τα βάρη και τις εισόδους.
        if len(float_outputs) > 0:
            return sum(t.sum() for t in float_outputs)
        else:
            # Fallback αν για κάποιο λόγο δεν βρεθεί κανένα float output
            return eeg.sum() * 0.0

        # Instantiate wrapper


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