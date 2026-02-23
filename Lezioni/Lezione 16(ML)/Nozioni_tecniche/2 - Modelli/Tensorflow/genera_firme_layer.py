import inspect
from pathlib import Path

def get_layers_module():
    # Prova prima Keras "standalone" (Keras 3)
    try:
        import keras
        from keras import layers
        return "keras", layers
    except Exception:
        # Fallback: TensorFlow Keras
        import tensorflow as tf
        return "tf.keras", tf.keras.layers

def is_layer_class(obj, base_layer):
    return isinstance(obj, type) and issubclass(obj, base_layer)

def main(out_path="KERAS_LAYERS_SIGNATURES.md"):
    prefix, layers_mod = get_layers_module()
    base = layers_mod.Layer

    items = []
    for name, obj in vars(layers_mod).items():
        if is_layer_class(obj, base):
            try:
                sig = str(inspect.signature(obj))
            except Exception:
                sig = "(signature non disponibile)"
            items.append((name, sig))

    items.sort(key=lambda x: x[0].lower())

    lines = [f"# Firme costruttori layer ({prefix})", ""]
    for name, sig in items:
        if prefix == "tf.keras":
            lines.append(f"## {name}")
            lines.append(f"`tf.keras.layers.{name}{sig}`")
        else:
            lines.append(f"## {name}")
            lines.append(f"`keras.layers.{name}{sig}`")
        lines.append("")

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"Creato: {out_path} (layer trovati: {len(items)})")

if __name__ == "__main__":
    main()