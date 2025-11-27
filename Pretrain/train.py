import torch
import time
from utils import AverageMeter, accuracy
from pruning import apply_mask_grads

def train_epoch(train_loader, model, criterion, optimizer, epoch, args, logger, masks=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # Compute output
        output = model(input)
        loss = criterion(output, target)

        # Measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        # Compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        
        if masks is not None:
            apply_mask_grads(model, masks)
            
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                        f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                        f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})')

    return top1.avg, losses.avg

def validate(val_loader, model, criterion, args, logger):
    # For FSL, validation should be episodic (Meta-Val) because classes are disjoint.
    # We reuse the meta_test logic but on the validation set.
    # Note: val_loader here is actually just a standard loader, but we need episodic sampling.
    # We will ignore val_loader and use the dataset directly or create a new episodic logic.
    # To avoid circular imports or duplication, we can import meta_test from evaluate.
    # However, to keep train.py independent, let's implement a lightweight meta_val here 
    # or better, just use meta_test from evaluate.py in main.py and remove this validate function.
    pass

