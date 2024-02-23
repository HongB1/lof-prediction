from tqdm import tqdm
from datetime import datetime
from torch.autograd import Variable
from sklearn.model_selection import StratifiedShuffleSplit, KFold
from torch.utils.data import Subset
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from EarlyStopping import EarlyStopping
import wandb
import pandas as pd
import torch
import sys
sys.path.append('/home/hb/lof/code')
sys.path.append('/home/hb/lof/')
from utils import LOFDataset
from torch_geometric.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

from LOFGCN.lofgcn import LOFGCN

norman_dataset = LOFDataset('norman')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train = pd.read_pickle("/data_hdd4/hb/lof/data/norman_train_info.pkl")

# device = torch.device('cpu')


kfold = KFold(n_splits=5, shuffle=True)

wandb.init(project='loa', entity="jeguring", reinit=True, )
project_name = f'{datetime.today().strftime("%m%d%H%M")}_norman_GCN'
wandb.run.name = project_name 

for fold, (train_idx, valid_idx) in enumerate(kfold.split(train.index.values)):

    globals()[f'{fold}_train_loss'] = []
    globals()[f'{fold}_train_precision'] = []
    globals()[f'{fold}_train_recall'] = []                          
    globals()[f'{fold}_train_f1'] = []
    globals()[f'{fold}_train_acc'] = []

    globals()[f'{fold}_valid_loss'] = []
    globals()[f'{fold}_valid_precision'] = []
    globals()[f'{fold}_valid_recall'] = []
    globals()[f'{fold}_valid_f1'] = []
    globals()[f'{fold}_valid_acc'] = []
    globals()[f'{fold}_lr'] = []

    globals()[f'{fold}_result'] = []
    print(f'FOLD {fold}')
    print('--------------------------------')
    model = LOFGCN(num_genes=10938, edge_index=norman_dataset[0].edge_index, hidden_size=128, num_classes=200)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=50, cycle_mult=2, max_lr=0.1, min_lr=0.000001, warmup_steps=20, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    criterion.to(device)
    best_model_weights = model.state_dict()
    best_loss = 1000000.0
    early_stopping = EarlyStopping(patience = 0.7, verbose = True)
    train_dataset = Subset(norman_dataset, train_idx)
    valid_dataset = Subset(norman_dataset, valid_idx)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=True)

    for epoch in tqdm(range(100)):
        print('-' * 60)
        print('Epoch {}/{}'.format(epoch+1, 100))
        train_loss = 0.0
        
        for _, batch_data in enumerate(tqdm(train_dataloader)):
            inputs = Variable(torch.tensor(batch_data.x).to(device, dtype=torch.float))

            labels = Variable(torch.tensor(batch_data.y).to(device)).float()
            model.train(True)
            pred = model(inputs)
            loss = criterion(pred, labels.float()).to(device)
            preds = (pred>0.5).float()

            '''backward'''
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            '''train record'''
            train_loss += loss.item()
            train_preds = (pred>=0.5).float()

            scheduler.step()
        '''epoch train record'''
        epoch_train_loss = train_loss / len(train_dataloader)

        with torch.no_grad():
            model.eval()

            valid_corrects = 0.0         
            valid_loss = 0.0
            valid_precision, valid_recall, valid_f1 = 0.0, 0.0, 0.0

            for i, batch_data in enumerate(tqdm(valid_dataloader, position=1, leave=True)):
                # model.train(False)
                inputs = Variable(torch.tensor(batch_data.x).to(device, dtype=torch.float))
                labels = Variable(torch.tensor(batch_data.y).to(device, )).float()
                pred = model(inputs) 
                loss = criterion(pred, labels.float()).to(device)

                '''valid record'''
                valid_loss += loss.item()
                valid_preds = (pred>=0.5).float()
        
        '''epoch valid record'''
        epoch_valid_loss = valid_loss / len(valid_dataloader) 
        globals()[f'{fold}_train_loss'].append(epoch_train_loss)
        globals()[f'{fold}_valid_loss'].append(epoch_valid_loss)

        if epoch_valid_loss < best_loss:
            best_loss = epoch_valid_loss
            best_model_weights = model.state_dict()
            
            checkpoint = {'epoch':epoch, 
            'loss':epoch_valid_loss,
                'model': model,
                        #'state_dict': model.module.state_dict(),
                            'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict()}
            torch.save(checkpoint, f"/home/hb/ptn/gnn/Graph-Travel/save_dir/{fold}fold_latest_epoch.pth")

        # Earlystopping & best 모델 저장
        savePath = "{}/{}fold_best_model.pth".format(wandb.config.save_dir, fold) 
        early_stopping(epoch_valid_loss, model, optimizer, savePath)
        if early_stopping.early_stop:
            print(f'Early stopping... fold:{fold} epoch:{epoch} loss:{epoch_valid_loss}')
            break

        wandb.log({f"{fold} fold train" : {"loss":epoch_train_loss}, f"{fold} fold val":{"loss":epoch_valid_loss} ,f"{fold} fold learning_rate":optimizer.param_groups[0]['lr']})
        globals()[f'{fold}_lr'].append(optimizer.param_groups[0]['lr'])
        scheduler.step(epoch_valid_loss) # reduced는 무조건 epoch에서 backward
        print("lr: ", optimizer.param_groups[0]['lr'])
        print('-' * 60)
        print()
    torch.cuda.empty_cache()

plt.plot(globals()['0_valid_loss'], label="0fold")
plt.plot(globals()['1_valid_loss'], label='1fold')
plt.plot(globals()['2_valid_loss'], label='2fold')
plt.plot(globals()['3_valid_loss'], label='3fold')
plt.plot(globals()['4_valid_loss'], label='4fold')
plt.title('Validation loss')
plt.xlabel("epoch")
plt.ylabel("Validation loss")
plt.legend()
plt.show()
save_dir = '/data_hdd4/hb/lof/model_record'
plt.savefig(save_dir + "/fig_saved.png")
wandb.run.save()
wandb.finish()

print('Best val Loss: {:4f}'.format(best_loss))
# load best model weightsQ
model.load_state_dict(best_model_weights)