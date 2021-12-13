# vq-wav2vec inference

A minimal code for [fairseq vq-wav2vec model](https://github.com/pytorch/fairseq/blob/main/examples/wav2vec/README.md#vq-wav2vec) inference. Runs without installing the fairseq toolkit and its dependencies.

#### Usage example:
```python
import torch
import fairseq
from models.wav2vec import Wav2VecModel

cp = torch.load('/path/to/vq-wav2vec.pt')
model = Wav2VecModel.build_model(cp['args'], task=None)
model.load_state_dict(cp['model'])
model.eval()

wav_input_16khz = torch.randn(1,10000)
z = model.feature_extractor(wav_input_16khz)
print(z[0].T.detach().numpy().shape) # output: (60, 512)
_, idxs = model.vector_quantizer.forward_idx(z)
print(idxs.shape) # output: torch.Size([1, 60, 2]), 60 timesteps with 2 indexes corresponding to 2 groups in the model
```
