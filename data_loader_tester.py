from gcommand_loader import GCommandLoader
import torch

dataset = GCommandLoader('/app/src/GCommandsPytorch/valid')

test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=None,
        num_workers=20, pin_memory=True, sampler=None)

for k, (input,label) in enumerate(test_loader):
    print(input.size(), len(label))
