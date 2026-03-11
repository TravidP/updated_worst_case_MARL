"""TensorFlow compatibility helpers for the legacy TF1-style codebase."""

try:
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
except ImportError:
    try:
        import tensorflow as tf  # type: ignore
    except ImportError as exc:
        raise ModuleNotFoundError(
            "TensorFlow is required. Install TensorFlow 2.15.x (or another TF build "
            "that still provides tf.compat.v1) to run this project."
        ) from exc

    if hasattr(tf, "compat") and hasattr(tf.compat, "v1"):
        tf = tf.compat.v1
        tf.disable_v2_behavior()
