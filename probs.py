import torch
probs = torch.load("probs.pt",weights_only=True).to('mps')
cpuProbs = probs.to('cpu')
cat = torch.distributions.Categorical(probs)
cpu_cat = torch.distributions.Categorical(cpuProbs)
print("entropy")
print(cat.entropy())
print(cpu_cat.entropy())
#always identical on my machine
print("mpu samples")
for i in range(10):
    print(cat.sample())
#(almost) always different, as expected for a somewhat uniform distribution
print('cpu samples')
for i in range(10):
    print(cpu_cat.sample())