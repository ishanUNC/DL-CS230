import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import argparse
import torchvision.models as models
from multitask_model import ResNetUNet, ResNetUNetTranspose
import tiramisu
from multitask_dataset import DVTDataset
from torch import nn
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import numpy as np
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
import matplotlib.pyplot as plt
import json

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

def evaluate(model, dataloader, criterion, writer = None, epoch = 0, val = False):
    auc = 0
    prec = 0
    recall = 0
    total_pr = np.array([])
    total_tpr = np.array([])
    total_threshold = np.array([])
    total_preds = np.array([])
    total_labels = np.array([])
    
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
            weights[mask] = 300
            bce = F.binary_cross_entropy_with_logits(segmentation_out.view(-1), segmentation.view(-1), weights.cuda())
            classification_out = torch.squeeze(classification_out)
            #print(classification_out)
            ce = criterion(classification_out, labels)
            loss = 0.5 * (0.25 * bce + 0.75 * dice) + 0.5 * (ce)
            outputs_copy = segmentation_out
            intersection = np.logical_and(segmentation.cpu().numpy(), outputs_copy.cpu().detach().numpy()).sum(1).sum(1)
            union = np.logical_or(segmentation.cpu().numpy(), outputs_copy.cpu().detach().numpy()).sum(1).sum(1)
            ious = intersection/union
            iou = np.mean(ious)
            if i % args.log_interval == 0:
                batch_number = i + epoch * len(dataloader)
                print("Val Batch Number: {} | Val Loss: {}  Dice: {}  BCE: {} IOU: {} CE:{}".format(batch_number, loss, dice, bce, iou, ce))
                writer.add_scalar("Validation BCE", bce, batch_number)
                writer.add_scalar("Validation CE", ce, batch_number)
                writer.add_scalar("Validation DICE", dice, batch_number)
                writer.add_scalar("Validation Loss", loss, batch_number)
                
        else:
            """
            classification_out = F.softmax(classification_out)
            preds = torch.max(classification_out, 1)[0]
            pr, tpr, thresholds = metrics.roc_curve(labels.cpu().numpy(), preds.cpu().detach().numpy())
            total_pr = np.concatenate((total_pr, pr))
            total_tpr = np.concatenate((total_tpr, tpr))
            total_threshold = np.concatenate((total_threshold, thresholds))
            auc = auc + roc_auc_score(labels.cpu().numpy(),preds.cpu().numpy())
            prec = prec + precision_score(labels.cpu().numpy(), preds.cpu().numpy())
            recall = recall + recall_score(labels.cpu().numpy(), preds.cpu().numpy())
            """
            try:
            # Compute AUC/ Precision
                #classification_out = torch.squeeze(classification_out)
                
                classification_out = F.softmax(classification_out)
                out_preds = torch.max(classification_out, 1)[0]
                total_preds = np.concatenate((total_preds, out_preds.cpu().detach().numpy()))
                total_labels = np.concatenate((total_labels, labels.cpu().numpy()))
                #pr, tpr, thresholds = metrics.roc_curve(labels.cpu().numpy(), preds.cpu().detach().numpy())
                #total_pr = np.concatenate((total_pr, pr))
                #total_tpr = np.concatenate((total_tpr, tpr))
                #total_threshold = np.concatenate((total_threshold, thresholds))
                preds = torch.argmax(classification_out, 1)
                auc = auc + roc_auc_score(labels.cpu().numpy(),preds.cpu().numpy())
                prec = prec + precision_score(labels.cpu().numpy(), preds.cpu().numpy())
                recall = recall + recall_score(labels.cpu().numpy(), preds.cpu().numpy())
                #print("AUC: ", roc_auc_score(labels.cpu().numpy(), preds.cpu().numpy()))
                #print("prec: ", precision_score(labels.cpu().numpy(), preds.cpu().numpy()))
                #print("recall: ", recall_score(labels.cpu().numpy(), preds.cpu().numpy()))
            except ValueError:
                pass
    if not val:
        auc = float(auc) / i
        prec = float(prec) / i
        recall = float(recall) / i
        print("Average AUC: ", auc)
        print("Average Prec: ", prec)
        print("Average Recall: ", recall)
        
        d = {'preds' : list(total_preds), 'labels' : list(total_labels)}
        jn = json.dumps(d)
        f = open("dict.json","w")
        f.write(jn)
        f.close()
        """
        plt.title('Multitask Model ROC for Classification')
        plt.plot(total_tpr, total_pr, 'b', label = 'AUC = %0.2f' % auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig('multitask_roc.png')
        """
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
    
    model = ResNetUNetTranspose(1)
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
    writer = SummaryWriter("multitask_final_runs")
    criterion_classification = nn.CrossEntropyLoss()
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
                #classification_out = torch.squeeze(classification_out)
                preds = torch.argmax(classification_out, 1)

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
                #print(torch.sum(F.sigmoid(segmentation_out)))

                bce = F.binary_cross_entropy_with_logits(segmentation_out.view(-1), segmentation.view(-1), weights.cuda())
                classification_out = torch.squeeze(classification_out)
                ce = criterion_classification(classification_out, labels)
                loss = 0.5 * (0.25 * bce + 0.75 * dice) + 0.5 * (ce)
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
                    writer.add_scalar("Training CE", ce, batch_number)
                    writer.add_scalar("Training DICE", dice, batch_number)
                    writer.add_scalar("Training Loss", loss, batch_number)
                    #print(torch.argmax(outputs, 1))
                    print("Batch Number: {} | Loss: {} BCE: {} DICE: {} IOU: {} CE: {}".format(batch_number,loss, bce, dice, iou, ce))
            torch.save(model.state_dict(), "multitask_final/multitask_model_{}.pt".format(epoch))
            val_dataset = DVTDataset(csv_file = args.val_csv_file, root_dir = args.root_dir, device = device)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 0, collate_fn = my_collate)
            evaluate(model, val_loader, criterion_classification, writer, epoch, val = True)
            if epoch % 3 == 0:
                scheduler.step()
    else:
        
        model.load_state_dict(torch.load(args.checkpoint_name + "multitask_final/multitask_model_20.pt"))
        #model.eval()
        test_dataset = DVTDataset(csv_file = args.test_csv_file, root_dir = args.root_dir, device = device)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 0, collate_fn = my_collate)

        evaluate(model, test_loader, nn.BCEWithLogitsLoss())
        
    #model.load_state_dict(torch.load(args.checkpoint_name + "model_0.pt"))
    #model.eval()
    #test_dataset = DVTDataset(csv_file = args.test_csv_file, root_dir = args.root_dir, device = device, collate_fn = my_collate)
    #test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 0, collate_fn = my_collate)
    #evaluate(model.eval(), test_loader, criterion)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv_file', default = 'train_femoral.csv')
    parser.add_argument('--val_csv_file', default = 'validation_femoral.csv')
    parser.add_argument('--test_csv_file', default = 'test_femoral.csv')
    parser.add_argument('--root_dir', default = '')
    parser.add_argument('--batch_size', default = 10)
    parser.add_argument('--epochs', default = 300)
    parser.add_argument('--lr', default = 0.0001)
    parser.add_argument('--log_interval', default = 10)
    parser.add_argument('--test', default = False)
    parser.add_argument('--checkpoint_name', default = "")
    
    
    args = parser.parse_args()
    main(args)

