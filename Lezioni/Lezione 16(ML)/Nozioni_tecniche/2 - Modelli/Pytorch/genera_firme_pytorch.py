import inspect
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

def collect_signatures(module, predicate):
    items = []
    for name, obj in vars(module).items():
        if predicate(obj):
            try:
                sig = str(inspect.signature(obj))
            except Exception:
                sig = "(signature non disponibile)"
            items.append((name, sig))
    return sorted(items, key=lambda x: x[0].lower())

def is_nn_module_class(obj):
    return isinstance(obj, type) and issubclass(obj, nn.Module)

def is_loss_class(obj):
    return isinstance(obj, type) and issubclass(obj, nn.modules.loss._Loss)

def is_optim_class(obj):
    return isinstance(obj, type) and issubclass(obj, optim.Optimizer)

def is_sched_class(obj):
    return isinstance(obj, type) and issubclass(obj, lrs.LRScheduler)

def main(out_path="PYTORCH_SIGNATURES.md"):
    lines = [f"# PyTorch signatures (torch {torch.__version__})", ""]

    for title, mod, pred, prefix in [
        ("nn.Module layers", nn, is_nn_module_class, "nn"),
        ("Loss functions (classes)", nn, is_loss_class, "nn"),
        ("Optimizers", optim, is_optim_class, "optim"),
        ("LR schedulers", lrs, is_sched_class, "lr_scheduler"),
    ]:
        lines += [f"## {title}", ""]
        items = collect_signatures(mod, pred)
        for name, sig in items:
            lines.append(f"- `{prefix}.{name}{sig}`")
        lines.append("")

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"Creato: {out_path}")

if __name__ == "__main__":
    main()