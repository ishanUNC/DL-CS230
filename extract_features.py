import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import argparse
import torchvision.models as models
from create_dataset import DVTDataset
from torch import nn
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import numpy as np
import csv


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def evaluate(model, dataloader, criterion, val=False):
    auc = 0
    prec = 0
    recall = 0
    for i, data in enumerate(dataloader):

        inputs, labels = data['image'], data['label']
        inputs = inputs.repeat(1, 3, 1, 1).float()
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        if val:
            # print(torch.argmax(outputs, 1))
            loss = criterion(outputs, labels)
            if i % args.log_interval == 0:
                print("Val Batch Number: {} | Val Loss: {}".format(i, loss))
        else:
            # Compute AUC/ Precision
            preds = torch.argmax(outputs, 1)
            # print(preds)
            auc = auc + roc_auc_score(labels.cpu().numpy(), preds.cpu().numpy())
            prec = prec + precision_score(labels.cpu().numpy(), preds.cpu().numpy())
            recall = recall + recall_score(labels.cpu().numpy(), preds.cpu().numpy())
            print("AUC: ", roc_auc_score(labels.cpu().numpy(), preds.cpu().numpy()))
            print("prec: ", precision_score(labels.cpu().numpy(), preds.cpu().numpy()))
            print("recall: ", recall_score(labels.cpu().numpy(), preds.cpu().numpy()))
    if not val:
        auc = float(auc) / i
        prec = float(prec) / i
        recall = float(recall) / i
        print("Average AUC: ", auc)
        print("Average Prec: ", prec)
        print("Average Recall: ", recall)
    return auc


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset = DVTDataset(csv_file=args.train_csv_file, root_dir=args.root_dir, device=device)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = models.vgg16()
    print("Loading model onto CUDA")
    model.load_state_dict(torch.load("vgg16-397923af.pth"))
    # model.classifier[6] = nn.Linear(4096,2)
    model.classifier[6] = Identity()
    for param in model.parameters():
        param.requires_grad = False

    model = model.cuda()
    model.train()

    print("Finished loading model onto CUDA")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # weights = np.zeros(1000)
    # weights[0] = 1.0
    # weights[1] = 1.0
    criterion = nn.CrossEntropyLoss()
    outs = {}

    if not args.test:
        for epoch in range(args.epochs):
            for i, data in enumerate(train_loader):
                if i > 20:
                    break
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels, segmentation_file = data['image'], data['label'], data["segmentation_file"]
                if segmentation_file[0] not in outs:
                    outs[segmentation_file[0]] = []
                # print(inputs.shape)
                inputs = inputs.repeat(1, 3, 1, 1).float()
                inputs, labels = inputs.cuda(), labels.cuda()
                # print(inputs.shape)
                # zero the parameter gradients
                optimizer.zero_grad()
                # print("Forward pass")
                # forward + backward + optimize

                outputs = model(inputs)
                outs[segmentation_file[0]].append(outputs)

                # print(torch.argmax(outputs, 1))
                # print("Forward pass complete.")
                # loss = criterion(outputs, labels)
                # loss.backward()
                # print("Backward pass.")
                # optimizer.step()
                if i % args.log_interval == 0:
                    print("Batch Number: {}".format(i))
                #    print("Batch Number: {} | Loss: {}".format(i,loss))

            # val_dataset = DVTDataset(csv_file = args.val_csv_file, root_dir = args.root_dir, device = device)
            # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 0)
            # evaluate(model, val_loader, criterion, val = True)
            # test_dataset = DVTDataset(csv_file = args.test_csv_file, root_dir = args.root_dir, device = device)
            # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 0)
            # evaluate(model, test_loader, criterion)
            # torch.save(model.state_dict(), "model_{}.pt".format(epoch))
    # model.load_state_dict(torch.load(args.checkpoint_name + "model_0.pt"))
    # model.eval()
    # test_dataset = DVTDataset(csv_file = args.test_csv_file, root_dir = args.root_dir, device = device)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 0)
    # evaluate(model.eval(), test_loader, criterion)
    means = {}
    for key in outs.keys():
        mean_encoding = torch.mean(torch.stack(outs[key]))
        means[key] = mean_encoding
    dists = []
    iterate = means.keys()
    minimums = {}
    for i1, mean1 in enumerate(iterate):
        minimum = 100000000000
        min_key = ""
        for i2, mean2 in enumerate(iterate):
            if mean1 != mean2:
                val = torch.dist(means[mean1], means[mean2], p=2).item()
                if val < minimum:
                    minimum = val
                    min_key = mean2
        minimums[mean1] = min_key
    print(minimums)

    """
    with open('outs.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in outs.items():
            writer.writerow([key, value])
    """


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv_file', default='train.csv')
    parser.add_argument('--val_csv_file', default='validation.csv')
    parser.add_argument('--test_csv_file', default='test.csv')
    parser.add_argument('--root_dir', default='')
    parser.add_argument('--batch_size', default=1)
    parser.add_argument('--epochs', default=1)
    parser.add_argument('--lr', default=0.00001)
    parser.add_argument('--log_interval', default=10)
    parser.add_argument('--test', default=False)
    parser.add_argument('--checkpoint_name', default="")

    args = parser.parse_args()
    main(args)