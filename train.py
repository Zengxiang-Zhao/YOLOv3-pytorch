import argparse
import time

import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

import test  # Import test.py to get mAP after each epoch
from models import *
from utils.datasets import *
from utils.utils import *

# Hyperparameters
# 0.861      0.956      0.936      0.897       1.51      10.39     0.1367    0.01057    0.01181     0.8409     0.1287   0.001028     -3.441     0.9127  0.0004841
hyp = {'k': 10.39,  # loss multiple
       'xy': 0.1367,  # xy loss fraction
       'wh': 0.01057,  # wh loss fraction
       'cls': 0.01181,  # cls loss fraction
       'conf': 0.8409,  # conf loss fraction
       'iou_t': 0.1287,  # iou target-anchor training threshold
       'lr0': 0.001028,  # initial learning rate
       'lrf': -3.441,  # final learning rate = lr0 * (10 ** lrf)
       'momentum': 0.9127,  # SGD momentum
       'weight_decay': 0.0004841,  # optimizer weight decay
       }


def train(
        cfg,
        weight_file,
        imgs_path,
        labels_path,
        img_size=416,
        epochs=273,  # 500200 batches at bs 64, dataset length 117263
        batch_size=16,
        transfer = False,
):


    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')

    # Initialize model
    model = Darknet(cfg, img_size).to(device)
    # load_darknet_weights(model,weight_file)

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=hyp['lr0'], momentum=hyp['momentum'], weight_decay=hyp['weight_decay'])

    cutoff = -1  # backbone reaches to cutoff layer
    start_epoch = 0
    best_loss = float('inf')
    nf = int(model.module_defs[model.yolo_layers[0] - 1]['filters'])  # yolo layer size (i.e. 255)
    
    # load the model
    if transfer:
        assert weight_file.endswith('.pt'), 'Please convert the weights to pytorch style'
        chkpt = torch.load(weight_file , map_location=device)
        model.load_state_dict({k: v for k, v in chkpt['model'].items() if v.numel() > 1 and v.shape[0] != 255},
                              strict=False) # only load state of conv witch not the former conv of yolo layer
        
        for p in model.parameters():
            p.requires_grad = True if p.shape[0] == nf else False
    else:
        if weight_file.endswith('.pt'):
            chkpt = torch.load(weight_file, map_location=device)
            model.load_state_dict(chkpt['model'])
        elif weight_file.endswith('.weights'): # YOLO original weights storage style
            load_darknet_weights(model,weight_file)
        else:
            print(f'The {weight_file} if not compatible for the model')

    # Scheduler https://github.com/ultralytics/yolov3/issues/238

    lf = lambda x: 1 - 10 ** (hyp['lrf'] * (1 - x / epochs))  # inverse exp ramp
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf, last_epoch=start_epoch - 1)

    # Dataset
    dataset = LoadImagesAndLabels(imgs_path, labels_path, img_size=img_size, augment=True)


    # Dataloader
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            # num_workers=opt.num_workers,
                            shuffle=True,
                            pin_memory=True,
                            collate_fn=dataset.collate_fn)

    # Start training
    t = time.time()
    model.hyp = hyp  # attach hyperparameters to model
    model_info(model)
    nb = len(dataloader)
    results = (0, 0, 0, 0, 0)  # P, R, mAP, F1, test_loss
    
    for epoch in range(start_epoch, epochs):
        model.train()
        print(('\n%8s%12s' + '%10s' * 7) % ('Epoch', 'Batch', 'xy', 'wh', 'conf', 'cls', 'total', 'nTargets', 'time'))

        mloss = torch.zeros(5).to(device)  # mean losses
        for i, (imgs, targets, _, _) in enumerate(dataloader):
            imgs = imgs.to(device)
            targets = targets.to(device)
            nt = len(targets)

            # Run model
            pred = model(imgs)

            optimizer.zero_grad()
            # Compute loss
            loss, loss_items = compute_loss(pred, targets, model)
            if torch.isnan(loss):
                print('WARNING: nan loss detected, ending training')
                return results

            loss.backward()
            optimizer.step()
            scheduler.step()

            # Update running mean of tracked metrics
            mloss = (mloss * i + loss_items) / (i + 1)

            # Print batch results
            s = ('%8s%12s' + '%10.3g' * 7) % (
                '%g/%g' % (epoch, epochs - 1),
                '%g/%g' % (i, nb - 1), *mloss, nt, time.time() - t)
            t = time.time()
            print(s)

        with torch.no_grad():
            results = test.test(cfg, data_cfg, batch_size=batch_size, img_size=img_size, model=model,
                                conf_thres=0.1)
        # Write epoch results
        with open('results.txt', 'a') as file:
            file.write(s + '%11.3g' * 5 % results + '\n')  # P, R, mAP, F1, test_loss

        # Update best loss
        test_loss = results[4]
        if test_loss < best_loss:
            best_loss = test_loss

        # # Save training results
        # save = True and not opt.nosave
        # if save:
        #     # Create checkpoint
        #     chkpt = {'epoch': epoch,
        #              'best_loss': best_loss,
        #              'model': model.module.state_dict() if type(
        #                  model) is nn.parallel.DistributedDataParallel else model.state_dict(),
        #              'optimizer': optimizer.state_dict()}

        #     # Save latest checkpoint
        #     torch.save(chkpt, latest)

        #     # Save best checkpoint
        #     if best_loss == test_loss:
        #         torch.save(chkpt, best)

        #     # Save backup every 10 epochs (optional)
        #     if epoch > 0 and epoch % 10 == 0:
        #         torch.save(chkpt, weights + 'backup%g.pt' % epoch)

        #     # Delete checkpoint
        #     del chkpt

    return results



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=273, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='size of each image batch')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--weights', type=str, help='file path to yolov3.weights')
    parser.add_argument('--img-size', type=int, default=416, help='pixels')
    parser.add_argument('--imgs_path', type=str, help='folder contain images')
    parser.add_argument('--labels_path', type=str, help='folder contain labels')
    parser.add_argument('--transfer', type=bool, help='Whether only train the yolo layers')
    opt = parser.parse_args()
    print(opt, end='\n\n')


    # Train
    results = train(
        cfg = opt.cfg,
        weight_file = opt.weights,
        imgs_path = opt.imgs_path,
        labels_path = opt.labels_path,
        img_size=opt.img_size,
        epochs=opt.epochs,
        batch_size=opt.batch_size,
        transfer=opt.transfer,
    )


