import comet_ml
from comet_ml.integration.pytorch import log_model

import optuna

import cProfile
import pstats

import os
import time
import torch
import random
import numpy as np
from tqdm import tqdm

import torch.multiprocessing as mp
import torch.distributed as dist

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

from utils.cometml_logger import create_experiment, log_experiment, log_distill_metrics, log_model_weights, plot_distribution
from utils import parse_args, setup_ddp, cleanup_ddp
from utils.metrics import initialize_metrics, log_metrics
from utils.early_stopping import EarlyStopping

from models.model_hub import get_model
from data_utils import create_data_loader

# Train and Test Functions
from utils.train_nn_utils import train_model as train_nn_model
from utils.train_nn_utils import test_model as test_nn_model
from utils.train_trad_utils import train_model as train_trad_model
from utils.train_trad_utils import test_model as test_trad_model
from utils.train_spiral_utils import train_model as train_spiral_model
from utils.train_spiral_utils import test_model as test_spiral_model
from utils.train_gcn_utils import train_model as train_gcn_model
from utils.train_gcn_utils import test_model as test_gcn_model
from utils.train_cmkd_utils import train_model as train_xmodal_model
from utils.train_cmkd_utils import test_model as test_xmodal_model
from utils.train_cmkd_utils import pred_model as pred_xmodal_model

model_functions = {
    "spiral_net": (train_spiral_model, test_spiral_model),
    "mdc_gcn":(train_gcn_model, test_gcn_model),
    "xmodal": (train_xmodal_model, test_xmodal_model),
    "net": (train_nn_model, test_nn_model),
    "vgg": (train_nn_model, test_nn_model),
    "vit": (train_nn_model, test_nn_model),
    "swin": (train_nn_model, test_nn_model),
    "mlp": (train_nn_model, test_nn_model),
    "rf": (train_trad_model, test_trad_model),
    "svm": (train_trad_model, test_trad_model)
}

def main_worker(rank, args):
    args.rank = rank
    
    if args.ddp:
        setup_ddp(args, rank)
        # Ensure each process only uses the assigned GPU
        torch.cuda.set_device(args.gpu[rank])
        args.device = torch.device(f"cuda:{args.gpu[rank]}")
                
        dist.barrier()
    
    if rank == 0:
        print(f"Created Exp: {rank}")
        experiment = create_experiment(args)
        
        print("Excluded DAI 22, 25, 28")
        print("Used only one View")
        args.num_views = 1
    else:
        experiment = None
    
    # create dataloaders
    train_dataloader, val_dataloader, test_dataloader, classes_dict  = create_data_loader(args) 
    
    args.classes = list(classes_dict.values())
    args.num_classes = len(args.classes)
       
    _, _, _ = main(args, experiment, [train_dataloader, val_dataloader, test_dataloader], rank)
        
    if args.ddp:
        cleanup_ddp()
    
    if rank == 0 and experiment is not None:
        experiment.end()
    
def main(args, experiment, dataloaders, rank):
    # load dataloaders
    train_dataloader, val_dataloader, test_dataloader = dataloaders
    
    if rank == 0 and args.dataset_name in ["rgb_images", "rgbd_images", "field_images"]:
        plot_distribution(args, experiment, train_dataloader, args.classes, mode="train")
        plot_distribution(args, experiment, val_dataloader, args.classes, mode="val")
        plot_distribution(args, experiment, test_dataloader, args.classes, mode="test")
    
        # get number of features
        inputs, _ = next(iter(train_dataloader)) 
        inputs = inputs.to(args.device, non_blocking=True) 
        args.input_channels = inputs.shape[1]
    else:
        inputs = 4
        args.input_channels = inputs
    
    # get model by name
    model = get_model(args, args.model_name)
    
    # Train Model
    if args.model_name in model_functions:
        train_model, test_model = model_functions[str(args.model_name)]
    elif any(keyword in args.model_name.lower() for keyword in ["net", "vgg", "vit", "swin"]):
        train_model, test_model = model_functions["net"]
        
    # Initialize metrics
    train_metrics, val_metrics, test_metrics = initialize_metrics(args)
    if args.model_name in ["xmodal", "efficientnet_b3"]:
        field_metrics = test_metrics.clone()
    
    # Early Stopping
    early_stopping = EarlyStopping(patience=25, min_delta=1e-4)
    
    print("Begin Training", flush=True)

    # Train/Fit the Models
    if args.model_name not in ["svm", "rf"]:
        criterion = torch.nn.CrossEntropyLoss()
        if args.model_name == "xmodal":
            params = list(model.student.parameters()) + list(model.student_proj.parameters())
        else:
            params = model.parameters()
        
        if args.optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5, factor=0.25) 
        
        # multi-gpu training
        if args.dataparallel:
            model = torch.nn.DataParallel(model)
        elif args.ddp:
            torch.cuda.set_device(rank)
            model.to(torch.device(f"cuda:{args.gpu[rank]}"))

            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[rank],
                output_device=rank,
            )
        elif args.model_name == "xmodal":
            model.teacher = model.teacher.to(args.device)  
            model.student = model.student.to(args.device)  
        else:
            model.to(args.device)   
        
            print(f"Model device check: {next(model.parameters()).device}")
        
        if args.ddp:
            dist.barrier() # synchronize before starting training
        
        epoch_pbar = tqdm(range(args.epochs), total=args.epochs, desc=f"Training Model (rank {args.rank})", unit="epoch") 
        for epoch in epoch_pbar:  

            if args.ddp:
                train_dataloader.sampler.set_epoch(epoch)  
                val_dataloader.sampler.set_epoch(epoch)                 
            
            # reset metrics for this epoch
            train_metrics.reset()
            val_metrics.reset()
            test_metrics.reset()
            if args.model_name in ["xmodal", "efficientnet_b3"]:
                field_metrics.reset()
                
            train_loss, train_acc, y_true_train, y_pred_train = train_model(
                args,
                model,
                optimizer,
                scheduler,
                criterion,
                train_dataloader,
                epoch,
                train_metrics,
                return_preds=True,
            )
            
            if rank == 0 and experiment is not None:
                if args.model_name == "xmodal":
                    train_loss, loss_ce, loss_kd, loss_feat = train_loss
                    log_distill_metrics(args, experiment, train_loss, loss_ce, loss_kd, loss_feat, epoch, mode="train")
                
                log_experiment(args, experiment, train_metrics, train_loss, epoch, y_true_train, y_pred_train, train_dataloader, model, mode="train")

                # Log learning rate for this epoch
                current_lr = optimizer.param_groups[0]['lr']
                experiment.log_metric(f"learning_rate", current_lr, step=epoch)
                        
            val_loss, val_acc, y_true, y_pred = test_model(
                args,
                model,
                optimizer,
                scheduler,
                criterion,
                val_dataloader,
                epoch,
                val_metrics,
                return_preds=True,
            )        

            if rank == 0 and experiment is not None:
                #if args.model_name == "xmodal":
                    #val_loss, loss_ce, loss_kd, loss_feat = val_loss
                    #log_distill_metrics(args, experiment, val_loss, loss_ce, loss_kd, loss_feat, epoch, mode="val")
                
                log_experiment(args, experiment, val_metrics, val_loss, epoch, y_true, y_pred, val_dataloader, model, mode="val")

                # Step lr scheduler
                scheduler.step(val_loss)                                 
                        
                epoch_pbar.set_postfix({"Train Loss": train_loss, "Val Loss": val_loss, "Val acc": val_acc})
                
                # Check Early Stopping
                if epoch > 1:
                    early_stopping(val_loss, model, epoch, experiment)
                    if early_stopping.early_stop:
                        if isinstance(val_acc, list):
                            acc_str = ", ".join([f"{a:.3f}" for a in val_acc])
                        else:
                            acc_str = f"{val_acc:.5f}"
                        print(
                            f"\nStopped at Epoch: {epoch} \tVal Accuracy: {acc_str} \tVal Loss: {val_loss:.5f}"
                        )
                        
                        model.load_state_dict(early_stopping.best_weights)
                        args.epochs = epoch + 1
                        break
        
        # Test model
        test_loss, test_acc, y_true, y_pred = test_model(
            args,
            model,
            optimizer,
            scheduler,
            criterion,
            test_dataloader,
            epoch,
            test_metrics,
            return_preds=True,
            task="predict",
        ) 
               
        if rank == 0 and experiment is not None:
            #if args.model_name == "xmodal":
            #    test_loss, loss_ce, loss_kd, loss_feat = test_loss
            #    log_distill_metrics(args, experiment, test_loss, loss_ce, loss_kd, loss_feat, epoch, mode="test")
            
            log_experiment(args, experiment, test_metrics, test_loss, epoch, y_true, y_pred, test_dataloader, model, mode="test")
                            
            acc_str = (", ".join([f"{a:.3f}" for a in test_acc])
                if isinstance(test_acc, list)
                else f"{test_acc:.5f}")
            
            print(f"\nFinished Training: \tTest Accuracy: {acc_str} \tTest Loss: {test_loss:.5f}")
            log_model_weights(args, experiment, model)

            if args.model_name in ["xmodal", "efficientnet_b3"]:
                if args.dataset_name == "rgbd_images":
                    old_conv = model.features[0][0]

                    new_conv = torch.nn.Conv2d(
                        in_channels=3,
                        out_channels=old_conv.out_channels,
                        kernel_size=old_conv.kernel_size,
                        stride=old_conv.stride,
                        padding=old_conv.padding,
                        bias=(old_conv.bias is not None)
                    )

                    # copy pretrained weights (first 3 channels)
                    with torch.no_grad():
                        new_conv.weight[:] = old_conv.weight[:, :3, :, :]

                    model.features[0][0] = new_conv.to(args.device)
                
                args.dataset_name = "field_images"
                _,_, field_dataloader, classes_dict = create_data_loader(args)
                plot_distribution(args, experiment, field_dataloader, args.classes, mode="pred")
                if args.model_name == "xmodal":
                    test_loss, test_acc, y_true, y_pred = pred_xmodal_model(
                        args, model, None, None, criterion,
                        field_dataloader, epoch, field_metrics,
                        return_preds=True, task="test"
                    )
                else:  # efficientnet_b3
                    test_loss, test_acc, y_true, y_pred = test_model(
                        args, model, None, None, criterion,
                        field_dataloader, epoch, field_metrics,
                        return_preds=True, task="test"
                    )
                    
                acc_str = (", ".join([f"{a:.3f}" for a in test_acc])
                    if isinstance(test_acc, list)
                    else f"{test_acc:.5f}")
                
                log_experiment(args, experiment, field_metrics, test_loss, epoch, y_true, y_pred, field_dataloader, model, mode="pred")
                print(f"\nField Image Predictions: \tAccuracy: {acc_str} \t Loss: {test_loss:.5f}")
    else:
        #TODO: set up traditional machine learning models
        train_metrics.reset()
        val_metrics.reset()
        test_metrics.reset()

        train_loss, train_acc, y_train, y_pred_train = train_model(args, model, train_dataloader, args.epochs - 1, train_metrics, return_preds=True)
        val_loss, val_acc, y_val, y_pred_val = test_model(args, model, val_dataloader, args.epochs - 1, val_metrics, return_preds=True, task="val")
        test_loss, test_acc, y_test, y_pred = test_model(args, model, test_dataloader, args.epochs - 1, test_metrics, return_preds=True, task="test")
        
        # Log experiments
        log_experiment(args, experiment, train_metrics, train_loss, args.epochs - 1, y_train, y_pred_train, train_dataloader, model, mode="train")
        log_experiment(args, experiment, val_metrics, val_loss, args.epochs - 1, y_val, y_pred_val, val_dataloader, model, mode="val")
        log_experiment(args, experiment, test_metrics, test_loss, args.epochs - 1, y_test, y_pred, test_dataloader, model, mode="test")

    print("End Training", flush=True)

    return train_loss, val_loss, test_loss


if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = parse_args("./default_config.yaml", desc="3d_images")
    
    # set random seeds
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed) 
    torch.backends.cudnn.enabled = False       # ensure no cuDNN dependency
    torch.backends.cuda.matmul.allow_tf32 = True  # speed up matmul on A100
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    
    #torch.backends.cudnn.deterministic = False
    #torch.backends.cudnn.benchmark = True
    
    #os.environ["CUDA_LAUNCH_BLOCKING"]="1"
    
    # set device
    if args.gpu and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu))
        
        if args.dataparallel or args.ddp:
            print(f"Multi-GPU enabled. Using GPUs: {args.gpu}")
        else:
            print(f"Single-GPU mode. Using GPU {args.gpu[0]}")
            torch.cuda.set_device(int(args.gpu[0]))
            args.device = torch.device(f"cuda:{args.gpu[0]}")
    else:
        args.device = torch.device("cpu")
        print("Using CPU.")
        
    if args.ddp:
        args.world_size = len(args.gpu)
        args.lr = args.lr * args.world_size
        args.batch_size = args.batch_size * args.world_size
        
        mp.spawn(main_worker, args=(args,), nprocs=args.world_size)
    else:
        # Single GPU or DataParallel logic
        #main_worker(0, args)    

        args.rank = 0
    
        print(f"Created Exp: {args.rank}")
        experiment = create_experiment(args)
        
        print("Excluded DAI 22, 25, 28")
        args.num_views =1

        # create dataloaders
        train_dataloader, val_dataloader, test_dataloader, classes_dict  = create_data_loader(args) 
        
        args.classes = list(classes_dict.values())
        args.num_classes = len(args.classes)
        
        _, _, _ = main(args, experiment, [train_dataloader, val_dataloader, test_dataloader], 0)
        
        experiment.end()
    
    
    torch.cuda.empty_cache()
