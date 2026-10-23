# Public-release sanitization record

The public tree was generated from the frozen research packages with these transformations:

1. Machine-specific absolute paths were replaced by environment placeholders.
2. Ten OR2 NPZ pixel exemplars containing third-party DIODE RGB/depth arrays were omitted. Their filenames and original SHA-256 values remain in the OR2 per-image JSON markers.
3. Raw datasets, upstream model repositories, model weights, credentials, logs, and temporary files were not copied.
4. A new SHA-256 manifest was generated over the sanitized public tree.

Numerical CSV records were not rounded or recomputed during sanitization.
