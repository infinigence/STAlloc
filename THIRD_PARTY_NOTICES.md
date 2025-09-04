# Third-Party Notices

This project is licensed under Apache-2.0. It includes or derives from third-party components under their respective licenses:

- PyTorch CUDACachingAllocator (BSD-3-Clause)
  - Files: `Allocator/CUDACachingAllocator.h`, `Allocator/CUDACachingAllocator.cpp`
  - Status: Derived and modified.
  - License: See `THIRD_PARTY_LICENSES/PYTORCH-BSD-3-CLAUSE.txt`.
  - Upstream: https://github.com/pytorch/pytorch

- LLVM MathExtras (Apache-2.0 WITH LLVM-exception)
  - File: `Allocator/llvmMathExtras.h`
  - Status: Copied with original license header preserved.
  - License: See `THIRD_PARTY_LICENSES/LLVM-Apache-2.0-with-LLVM-exception.txt`.
  - Upstream: https://github.com/llvm/llvm-project

- ska flat_hash_map (Boost Software License 1.0)
  - File: `Allocator/flat_hash_map.h`
  - Status: Copied with original license header preserved.
  - License: See `THIRD_PARTY_LICENSES/BOOST-1.0.txt`.
  - Upstream: https://github.com/skarupke/flat_hash_map

Notes on modifications:
- We have adjusted `CUDACachingAllocator.*` to fit this project and added headers indicating derivation and SPDX identifiers.
- Other third-party files retain their original license headers as required. 