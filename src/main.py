
import numpy as np
import torch
import tqdm
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
from utils import seed_everything
from resnet import ResNet18
from cifar10 import cifar10_dataset

def train(model, train_loader , val_loader, args,criterion,optimizer,mixup=False,alpha = 1):
    """
       Simple training loop for PyTorch model.
       args:  epochs , model_path='model.ckpt' ,load_model=False, min_val_acc_to_save=88.0

    """ 
    writer=SummaryWriter(os.path.jooin(args.save_path,'tensorboard',args.exp_name))

    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    best_val_acc=0
    # Make sure model is in training mode.
    if args.load_model and args.load_path:
      print('Loading the model from ckpt.....')
      train_ckpt=torch.load(args.load_path)
      model.load_state_dict(train_ckpt['model_state_dict'])
      print('The model is ready!')

    model.train()
    optimizer.zero_grad()

    # Move model to the device (CPU or GPU).
    model.to(device)
    
    # Exponential moving average of the loss.
    ema_loss = None
    losses=[]
    val_losses = []
    train_accs=[]
    val_accs=[]
    CEL = criterion

    print(f'----- Training on {device} -----')
    # Loop over epochs.
    for epoch in range(args.epochs):
        correct = 0
        num_examples=0
        ema_loss = 0
        # Loop over data.
        loop=tqdm(enumerate(train_loader , start =epoch*len(train_loader)),leave=False, total=len(train_loader))
        for step , (images, target) in loop:
            if args.mixup:
              lam = np.random.beta(alpha, alpha)
              ids = torch.randperm(images.shape[0])
              x = (lam * images + (1. - lam) * images[ids]).to(device)
              # Forward pass
              output = model(x)

              loss = lam * CEL(output, target.to(device)) + (1-lam) * CEL(output, target[ids].to(device))
            else:
              # Forward pass.
              output = model(images.to(device))
              loss = CEL(output.to(device), target.to(device))

            # Backward pass.
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


            # NOTE: It is important to call .item() on the loss before summing.
            ema_loss += (loss.item() - ema_loss) * 0.01 
            # Compute the correct classifications
            preds = output.argmax(dim=1, keepdim=True)
            correct+= preds.cpu().eq(target.view_as(preds)).sum().item()
            num_examples+= images.shape[0]
            train_acc=correct/num_examples
            #tqdm
            loop.set_description(f"Epoch [{epoch+1}/{args.epochs}]")
            loop.set_postfix(loss=ema_loss, acc=train_acc)
        
        #write the loss to tensorboard    
        writer.add_scalar('train loss', ema_loss, global_step=epoch)
        writer.add_scalar('train acc', train_acc, global_step=epoch)
        writer.add_scalar('train error', 1-train_acc, global_step=epoch)

        losses.append(ema_loss)
        train_accs.append(train_acc)

        val_acc, val_loss= test(model ,val_loader, device,CEL)
        val_losses.append(val_loss)
        writer.add_scalar('val loss', val_loss, global_step=epoch)
        writer.add_scalar('val acc', val_acc, global_step=epoch)
        writer.add_scalar('val eror', 1-val_acc, global_step=epoch)

        val_accs.append(val_acc)
        if val_acc > best_val_acc and val_acc > args.min_val_acc_to_save:
            print(f'validation accuracy increased from {best_val_acc} to {val_acc}  , saving the model ....')
            #saving training ckpt
            chk_point={'model_sate_dict':model.state_dict(), 'epochs':epoch+1, 'best_val_acc':best_val_acc}
            torch.save(chk_point, os.path.join(args.save_path,args.exp_name,model.ckpt))
            best_val_acc=val_acc
        print('-------------------------------------------------------------')
        
        if epoch+1==100 or epoch+1==150:
            optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']/10

    return train_accs , val_accs, losses, val_losses
    
def test(model, data_loader, device,criterion):
    """Measures the accuracy of a model on a data set.""" 
    # Make sure the model is in evaluation mode.
    model.eval()
    correct = 0
    total = 0
    ema_loss = 0
    print(f'----- Model Evaluation on {device}-----')
    # We do not need to maintain intermediate activations while testing.
    with torch.no_grad():
        
        # Loop over test data.
        for features, targets in data_loader:
            features, targets = features.to(device), targets.to(device)
  
            # Forward pass.
            outputs = model(features)
            
            # Get the label corresponding to the highest predicted probability.
            preds = outputs.argmax(dim=1, keepdim=True) #[bs x 1]
            
            # Count number of correct predictions.
            correct += preds.eq(targets.view_as(preds)).sum().item()

            loss = criterion(outputs , targets)
            ema_loss  +=  (loss.item() - ema_loss) * 0.01 

    model.train()
    # Print test accuracy.
    percent = 100. * correct / len(data_loader.sampler)
    print(f'validation accuracy: {correct} / {len(data_loader.sampler)} ({percent:.2f}%)')
    return percent , ema_loss


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Mixup Training')
    parser.add_argument('--mixup',default=True)

    parser.add_argument('--epochs', default=1, type=int, help='number of total epochs to run')
    parser.add_argument('--batch-size', default=128, type=int, help='mini-batch size')
    parser.add_argument('--learning-rate', default=0.1, type=float,help='base learning rate')
   
    parser.add_argument('--min-val-acc-to-save', default=30.0, type=float )

    parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay')

    parser.add_argument('--load-model',default=False)
    parser.add_argument('--load-path', default = '', type=str)
    parser.add_argument('--save-path', default = '/gdrive/MyDrive/simsiam/', type=str)

    parser.add_argument('--exp-name', default = 'resnet18_mixup', type=str)
    parser.add_argument('--seed', default=123, type=int)


    args=parser.parse_args()

    seed_everything(args.seed)
    model = ResNet18()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                      momentum=0.9, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
 
    train_loader , val_loader = cifar10_dataset(args.batch_size)

    train(model, train_loader, val_loader, args,criterion,optimizer)