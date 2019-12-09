import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import argparse
import torchvision.models as models
from classification_dataset import DVTDataset
from torch import nn
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import numpy as np
import re
from torch.utils.tensorboard import SummaryWriter
import json


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def evaluate(model, dataloader, criterion, writer=None, epoch=0, val=False):
    auc = 0
    prec = 0
    recall = 0
    model = model.eval()
    total_preds = np.array([])
    total_labels = np.array([])
    total_hard_preds = np.array([])

    for i, data in enumerate(dataloader):
        inputs, labels, access_number = data['image'], data['label'], data['access_number']
        inputs = inputs.repeat(1, 3, 1, 1).float()
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        if val:
            # print(torch.argmax(outputs, 1))
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, 1)
            # print(preds)
            # auc = auc + roc_auc_score(labels.cpu().numpy(), preds.cpu().numpy())
            # prec = prec + precision_score(labels.cpu().numpy(), preds.cpu().numpy())
            # recall = recall + recall_score(labels.cpu().numpy(), preds.cpu().numpy())
            # print("AUC: ", roc_auc_score(labels.cpu().numpy(), preds.cpu().numpy()))
            # print("prec: ", precision_score(labels.cpu().numpy(), preds.cpu().numpy()))
            # print("recall: ", recall_score(labels.cpu().numpy(), preds.cpu().numpy()))
            if i % args.log_interval == 0:
                batch_number = i + epoch * len(dataloader)
                print("Val Batch Number: {} | Val Loss: {} ".format(batch_number, loss))
                writer.add_scalar("Validation Loss", loss, batch_number)

        else:
            # Compute AUC/ Precision
            preds = torch.argmax(outputs, 1)
            # soft = nn.Softmax()
            # out = soft(outputs)
            # print(preds)
            try:
                # print("Labels", labels)
                # print("Preds", preds)
                out_preds = torch.max(outputs, 1)[0]
                out_hard_preds = torch.argmax(outputs, 1)
                total_hard_preds = np.concatenate((total_hard_preds, out_hard_preds.cpu().detach().numpy()))
                total_preds = np.concatenate((total_preds, out_preds.cpu().detach().numpy()))
                total_labels = np.concatenate((total_labels, labels.cpu().numpy()))
                current_prec = precision_score(labels.cpu().numpy(), preds.cpu().numpy())
                current_recall = recall_score(labels.cpu().numpy(), preds.cpu().numpy())
                # print("Current Prec", current_prec)
                # print("Current Recall", current_recall)
                # print("Equal", current_prec == 1.0)
                auc = auc + roc_auc_score(labels.cpu().numpy(), preds.cpu().detach().numpy())
                prec = prec + precision_score(labels.cpu().numpy(), preds.cpu().numpy())
                recall = recall + recall_score(labels.cpu().numpy(), preds.cpu().numpy())

            except ValueError:
                pass
            # print("AUC: ", roc_auc_score(labels.cpu().numpy(), preds.cpu().numpy()))
            # print("prec: ", precision_score(labels.cpu().numpy(), preds.cpu().numpy()))
            # print("recall: ", recall_score(labels.cpu().numpy(), preds.cpu().numpy()))

    auc = float(auc) / i
    prec = float(prec) / i
    recall = float(recall) / i
    print("Average AUC: ", auc)
    print("Average Prec: ", prec)
    print("Average Recall: ", recall)
    d = {'preds': list(total_preds), 'labels': list(total_labels), 'hard_preds': list(total_hard_preds)}
    jn = json.dumps(d)
    f = open("dict_2.json", "w")
    f.write(jn)
    f.close()
    # writer.add_scalar("Validation Loss", loss, i)
    # writer.add_scalar("Validation AUC", auc, i)
    # writer.add_scalar("Validation Precision", prec, i)
    # writer.add_scalar("Validation Recall", recall, i)
    return auc


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset = DVTDataset(csv_file=args.train, root_dir=args.root_dir, device=device)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = models.densenet161()
    print("Loading model onto CUDA")
    state_dict = torch.load("densenet161-8d451a50.pth")
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)
    model.classifier = nn.Linear(2208, 2)

    model = model.cuda()
    model.train()

    print("Finished loading model onto CUDA")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = optim.Adam(model.parameters(), lr = 0.0001)
    # optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    writer = SummaryWriter("classification_lowlr_runs")
    # weights = np.zeros(1000)
    # weights[0] = 1.0
    # weights[1] = 1.0
    criterion = nn.CrossEntropyLoss()
    if not args.evaluate:
        for epoch in range(args.epochs):
            for i, data in enumerate(train_loader):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data['image'], data['label']
                # print(inputs.shape)
                inputs = inputs.repeat(1, 3, 1, 1).float()
                inputs, labels = inputs.cuda(), labels.cuda()
                # print(inputs.shape)
                # zero the parameter gradients
                optimizer.zero_grad()
                # print("Forward pass")
                # forward + backward + optimize

                outputs = model(inputs)
                # print(outputs)
                # print(torch.argmax(outputs, 1))
                # print("Forward pass complete.")
                loss = criterion(outputs, labels)
                loss.backward()
                # print("Backward pass.")
                optimizer.step()
                if i % args.log_interval == 0:
                    batch_number = i + epoch * len(train_loader)
                    # print(torch.argmax(outputs, 1))
                    writer.add_scalar("Training Loss", loss, batch_number)
                    print("Batch Number: {} | Loss: {}".format(i, loss))
            torch.save(model.state_dict(), "classification_lowlr/model_{}.pt".format(epoch))
            val_dataset = DVTDataset(csv_file=args.val, root_dir=args.root_dir, device=device)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                                     num_workers=0)
            evaluate(model, val_loader, criterion, writer, epoch, val=True)
            # scheduler.step()
            # test_dataset = DVTDataset(csv_file = args.test, root_dir = args.root_dir, device = device)
            # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 0)
            # evaluate(model, test_loader, criterion)

    model.load_state_dict(torch.load(args.checkpoint_name + "classification_dropout/model_9.pt"))
    model.eval()
    test_dataset = DVTDataset(csv_file=args.test, root_dir=args.root_dir, device=device)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    evaluate(model.eval(), test_loader, criterion)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='train_femoral.csv')
    parser.add_argument('--val', default='validation_femoral.csv')
    parser.add_argument('--test', default='train_femoral.csv')
    parser.add_argument('--root_dir', default='')
    parser.add_argument('--batch_size', default=10)
    parser.add_argument('--epochs', default=10)
    parser.add_argument('--lr', default=0.001)
    parser.add_argument('--log_interval', default=10)
    parser.add_argument('--val_interval', default=100)
    parser.add_argument('--evaluate', default=False)
    parser.add_argument('--checkpoint_name', default="")

    args = parser.parse_args()
    main(args)