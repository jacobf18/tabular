# Local baseline compatibility patches

These patches preserve the local baseline fixes without creating unpublished
commits in third-party submodules. They are separate from MatrixFormer inference.

| Submodule | Recorded upstream commit | Patch |
|---|---|---|
| ReMasker | `69b410819c0d096fab111bb8b0e54c1edd4efcc2` | `remasker_compat.patch` |
| DiffPuter | `2fa55373655b9e910146d94820fc1012da0dfd75` | `diffputer_remasker_compat.patch` |

The fixes remove an unused Tkinter import and replace the removed NumPy
`np.float` alias with `np.float64` in ReMasker's positional embedding code.

From the repository root, on clean submodules at the recorded revisions:

```bash
git submodule update --init mcpfn/benchmark/remasker mcpfn/benchmark/DiffPuter
git -C mcpfn/benchmark/remasker apply --check ../patches/remasker_compat.patch
git -C mcpfn/benchmark/remasker apply ../patches/remasker_compat.patch
git -C mcpfn/benchmark/DiffPuter apply --check ../patches/diffputer_remasker_compat.patch
git -C mcpfn/benchmark/DiffPuter apply ../patches/diffputer_remasker_compat.patch
```

An already-patched checkout passes `git apply --reverse --check` instead. Leave
existing local modifications intact; these commands are intended for a clean
reproduction checkout. Applying the patches intentionally leaves the submodules
dirty, while their upstream commit references remain unchanged.
