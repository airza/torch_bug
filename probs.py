import torch
print(torch.__version__)
probs = torch.load("probs.pt",weights_only=True).to('mps')
cpuProbs = torch.load("probs.pt",weights_only=True).to('cpu')
cat = torch.distributions.Categorical(probs)
cpu_cat = torch.distributions.Categorical(cpuProbs)
for i in range(10):
    print(cat.sample())
for i in range(10):
    print(cpu_cat.sample())