# Third-Party Notices

This repository distributes and/or builds upon third-party open-source software.
The components below are **vendored** (their source is redistributed in this
repository) and remain governed by their own upstream licenses, which are
included alongside the source. NVIDIA modifications to these components are
licensed under Apache-2.0 and carry a per-file attribution header that preserves
the upstream copyright and license.

Additional third-party packages are fetched at build/runtime via the
`requirements.txt` / package manifests in each microservice and are **not**
redistributed here; they remain under their respective upstream licenses.

---

## DDM-Net (Generic Event Boundary Detection)

- **Upstream project:** https://github.com/MCG-NJU/DDM
- **Copyright:** Copyright (c) 2021 Mike Zheng Shou
- **License:** MIT License
- **Vendored locations in this repository:**
  - `microservices/sop-training-bp/microservices/ddm-training-ms/ddm/`
  - `microservices/sop-training-bp/microservices/evaluation-ms/ddm/`
- **Full license text:** see the `LICENSE` file in each of the directories above.

```
MIT License

Copyright (c) 2021 Mike Zheng Shou

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## TAO PyTorch Backbone (v2)

- **Copyright:** Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
- **License:** Apache License 2.0
- **Vendored locations in this repository:**
  - `microservices/sop-training-bp/microservices/ddm-training-ms/ddm/DDM-Net/modeling/tao_pytorch_backbone_v2/`
  - `microservices/sop-training-bp/microservices/evaluation-ms/ddm/DDM-Net/modeling/tao_pytorch_backbone_v2/`
- **Full license text:** see the `LICENSE` file in each of the directories above, and
  https://www.apache.org/licenses/LICENSE-2.0.
