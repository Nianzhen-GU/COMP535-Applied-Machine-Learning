from dataset import TrainData
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import model
import sys

if __name__ == "__main__":
    train_set = TrainData()
    train_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=False)
    print('loaded {} images'.format(len(train_loader)))

    torch.manual_seed(1)

    # Build model
    print("===> building net")

    model = model.Net()
    print(model)

    # optimizer and loss logger
    print("===> setting optimizer")
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    lossFn = nn.CrossEntropyLoss()

    print("===> training")
    for epoch in range(0, 30):
        model.train()
        loss_count = 0.
        for i, batch in enumerate(train_loader, 1):
            data = batch[0] 
            label = batch[1]

            optimizer.zero_grad()

            result = model(data)

            loss = lossFn(result, label.reshape(-1, 36))

            print(loss)
            loss_count += loss.item()

            
            loss.backward()
            optimizer.step()
        print(loss_count / len(train_loader))
