import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import argparse
import torchvision.models as models
from multitask_model import ResNetUNet, ResNetUNetSegmentation
import tiramisu
from multitask_dataset import DVTDataset
from torch import nn
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import numpy as np
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
def my_collate(batch):
    pre_len = len(batch)
    batch = list(filter(lambda x:x is not None, batch))
    diff = len(batch) - pre_len
    for i in range(diff):
        batch.append(batch[-1])
    return default_collate(batch)

def evaluate(model, dataloader, criterion, writer, epoch = 0, val = False):
    auc = 0
    prec = 0
    recall = 0
    for i, data in enumerate(dataloader):
        inputs, labels, segmentation = data['image'], data['label'], data["segmentation"]
        inputs = inputs.repeat(1,3,1,1).float()
        inputs, labels, segmentation = inputs.cuda(), labels.cuda(), segmentation.cuda()
        segmentation_out, classification_out = model(inputs)
        if val:
            dice = dice_loss(F.sigmoid(segmentation_out), segmentation)
            #print(outputs.view(-1).shape)
            #print(segmentation.view(-1).shape)
            mask = torch.nonzero(segmentation.view(-1))
            weights = torch.ones(segmentation.view(-1).shape)
            weights[mask] = 100
            bce = F.binary_cross_entropy_with_logits(segmentation_out.view(-1), segmentation.view(-1), weights.cuda())
            #classification_out = torch.squeeze(classification_out)
            #ce = criterion(classification_out.float(), labels.float())
            loss = 0.25 * bce + 0.75 * dice
            outputs_copy = segmentation_out
            intersection = np.logical_and(segmentation.cpu().numpy(), outputs_copy.cpu().detach().numpy()).sum(1).sum(1)
            union = np.logical_or(segmentation.cpu().numpy(), outputs_copy.cpu().detach().numpy()).sum(1).sum(1)
            ious = intersection/union
            iou = np.mean(ious)
            if i % args.log_interval == 0:
                batch_number = i + epoch * len(dataloader)
                print("Val Batch Number: {} | Val Loss: {}  Dice: {}  BCE: {} IOU: {}".format(i, loss, dice, bce, iou))
                writer.add_scalar("Validation BCE", bce, batch_number)
                #writer.add_scalar("Validation CE", ce, i)
                writer.add_scalar("Validation DICE", dice, batch_number)
                writer.add_scalar("Validation Loss", loss, batch_number)
                
        else:
            try:
            # Compute AUC/ Precision
                preds = torch.argmax(outputs, 1)
                #print(preds)
                auc = auc + roc_auc_score(labels.cpu().numpy(), preds.cpu().numpy())
                prec = prec + precision_score(labels.cpu().numpy(), preds.cpu().numpy())
                recall = recall + recall_score(labels.cpu().numpy(), preds.cpu().numpy())
                print("AUC: ", roc_auc_score(labels.cpu().numpy(), preds.cpu().numpy()))
                print("prec: ", precision_score(labels.cpu().numpy(), preds.cpu().numpy()))
                print("recall: ", recall_score(labels.cpu().numpy(), preds.cpu().numpy()))
            except ValueError:
                pass
    if not val:
        auc = float(auc) / i
        prec = float(prec) / i
        recall = float(recall) / i
        print("Average AUC: ", auc)
        print("Average Prec: ", prec)
        print("Average Recall: ", recall)
    return auc


def dice_loss(pred, target, smooth = 1.):

    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    #print(pred.shape)
    #print(target.shape)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset = DVTDataset(csv_file = args.train_csv_file, root_dir = args.root_dir, device = device)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 0, collate_fn = my_collate)
    
    model = ResNetUNetSegmentation(1)
    print("Loading model onto CUDA")
    #model.load_state_dict(torch.load("vgg16-397923af.pth"))
    #model.classifier[6] = nn.Linear(4096,2)
    #model = tiramisu.FCDenseNet57(n_classes=1)
    model = model.cuda()
    model.train()
    
    print("Finished loading model onto CUDA")

    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    #weights = np.zeros(1000)
    #weights[0] = 1.0
    #weights[1] = 1.0
    writer = SummaryWriter("segmentation_runs")
    criterion_classification = nn.BCEWithLogitsLoss()
    if not args.test:
        for epoch in range(args.epochs):
            for i, data in enumerate(train_loader):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels, segmentation = data['image'], data['label'], data["segmentation"]
                #print(inputs.shape)
                inputs = inputs.repeat(1,3,1,1).float()
                inputs, labels, segmentation = inputs.cuda(), labels.cuda(), segmentation.cuda()
                #print(inputs.shape)
                # zero the parameter gradients
                optimizer.zero_grad()
                # print("Forward pass")
                # forward + backward + optimize
                
                segmentation_out, classification_out = model(inputs)
                
                #print(torch.argmax(outputs, 1))
                #print("Forward pass complete.")
                #loss = criterion(outputs, labels)
                #print(F.sigmoid(outputs).sum())
                dice = dice_loss(F.sigmoid(segmentation_out), segmentation)
                #print(outputs.view(-1).shape)
                #print(segmentation.view(-1).shape)
                mask = torch.nonzero(segmentation.view(-1))
                weights = torch.ones(segmentation.view(-1).shape)
                weights[mask] = 100


                bce = F.binary_cross_entropy_with_logits(segmentation_out.view(-1), segmentation.view(-1), weights.cuda())
                #classification_out = torch.squeeze(classification_out)
                #ce = criterion_classification(classification_out.float(), labels.float())
                loss = 0.25 * bce + 0.75 * dice
                outputs_copy = segmentation_out
               
                intersection = np.logical_and(segmentation.cpu().numpy(), outputs_copy.cpu().detach().numpy()).sum(1).sum(1)
                union = np.logical_or(segmentation.cpu().numpy(), outputs_copy.cpu().detach().numpy()).sum(1).sum(1)
                ious = intersection/union
                iou = np.mean(ious)
                loss.backward()
                #print("Backward pass.")
                optimizer.step()
                if i % args.log_interval == 0:
                    batch_number = i + epoch * len(train_loader)
                    writer.add_scalar("Training BCE", bce, batch_number)
                    #writer.add_scalar("Training CE", ce, i)
                    writer.add_scalar("Training DICE", dice, batch_number)
                    writer.add_scalar("Training Loss", loss, batch_number)
                    #print(torch.argmax(outputs, 1))
                    print("Batch Number: {} | Loss: {} BCE: {} DICE: {} IOU: {}".format(batch_number,loss, bce, dice, iou))
            torch.save(model.state_dict(), "segmentation/segmentation_model_{}.pt".format(epoch))
            val_dataset = DVTDataset(csv_file = args.val_csv_file, root_dir = args.root_dir, device = device)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 0, collate_fn = my_collate)
            evaluate(model, val_loader, criterion_classification, writer, epoch, val = True)
            scheduler.step()
            #test_dataset = DVTDataset(csv_file = args.test_csv_file, root_dir = args.root_dir, device = device)
            #test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 0)
            #evaluate(model, test_loader, criterion)
            
    model.load_state_dict(torch.load(args.checkpoint_name + "model_0.pt"))
    model.eval()
    test_dataset = DVTDataset(csv_file = args.test_csv_file, root_dir = args.root_dir, device = device, collate_fn = my_collate)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 0, collate_fn = my_collate)
    evaluate(model.eval(), test_loader, criterion)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv_file', default = 'train_femoral.csv')
    parser.add_argument('--val_csv_file', default = 'validation_femoral.csv')
    parser.add_argument('--test_csv_file', default = 'test_femoral.csv')
    parser.add_argument('--root_dir', default = '')
    parser.add_argument('--batch_size', default = 10)
    parser.add_argument('--epochs', default = 300)
    parser.add_argument('--lr', default = 0.001)
    parser.add_argument('--log_interval', default = 10)
    parser.add_argument('--test', default = False)
    parser.add_argument('--checkpoint_name', default = "")
    
    
    args = parser.parse_args()
    main(args)