# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn

import time
import torch
from torch.optim import Adam

from settings import DATA_PATH, MODEL_PATH

from src.tools.easy_tools import initialize_logger, terminate_logger
from src.models.baselines import AlbertLargeFinetunedRACE, BertLargeFinetunedRACE


def train_baselines(args,
                    baseline_class,
                    data_name,
                    plm_name,
                    ckpt_path,
                    ):
    time_string = time.strftime("%Y-%m-%d-%H-%M-%S")
    logger = initialize_logger(filename=f"train-baseline-{time_string}.log", mode='w')
    logger.info(f"Arguments: {vars(args)}")
    logger.info(f"Using {args.device}")
    logger.info(f"Available devices: {torch.cuda.device_count()}")

    logger.info("Load model ...")
    model = eval(baseline_class)(pretrained_model_name_or_path=PLM_PATH[plm_name],
                                 device=args.device,
                                 dropout_rate=args.dropout_rate,
                                 max_length=args.max_length,
                                 )
    optimizer = eval(args.optimizer)(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_multiplier)

    train_logging = {"epoch": list(),
                     "iteration": list(),
                     "loss": list(),
                     "accuracy": list(),
                     }
    dev_logging = {"epoch": list(), "accuracy": list()}

    # Load checkpoint
    current_epoch = 0
    if ckpt_path is not None:
        logger.info(f"Load checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=torch.device(DEVICE))
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step_lr_scheduler.load_state_dict(checkpoint["scheduler"])
        current_epoch = checkpoint["epoch"] + 1
        train_logging = checkpoint["train_logging"]
        dev_logging = checkpoint["dev_logging"]
    logger.info(f"Current epoch: {current_epoch}")

    for epoch in range(args.n_epochs):
        # Train
        model.train()
        data_path = DATA_PATH[data_name]
        data_name_lower = data_name.lower()
        kwargs = {f"{data_name_lower}_path": data_path,
                  "batch_size": args.train_batch_size,
                  "types": ["train"],
                  "difficulties": ["high", "middle"],
                  }
        for iteration, train_batch_data in enumerate(eval(f"yield_{data_name_lower}_batch")(**kwargs)):
            loss = model(train_batch_data, mode="train")
            optimizer = zero_grad()
            loss.backward()
            optimizer.step()
            logger.info(f"Epoch: {epoch} - Iteration: {iteration} - Loss: {loss.item()}")
            training_logging["epoch"].append(epoch)
            training_logging["iteration"].append(iteration)
            training_logging["loss"].append(loss)
        # Validate
        model.eval()
        with torch.no_grad():
            kwargs = {f"{data_name_lower}_path": data_path,
                      "batch_size": args.dev_batch_size,
                      "types": ["dev"],
                      "difficulties": ["high", "middle"],
                      }
            for i, dev_batch_data in enumerate(eval(f"yield_{data_name_lower}_batch")(**kwargs)):
                metric = model(dev_batch_data, mode="dev")
                logger.info(metric)
    # Export train and dev logging
    train_logging_dataframe = pandas.DataFrame(train_logging, columns=list(train_logging.keys()))
    train_logging_dataframe.to_csv(os.path.join(LOGGING_DIR, f"{timestring}-{dataset_name}-{model_name}-{mode}.csv"),
                                   header=True, index=False, sep='\t')
    dev_logging_dataframe = pandas.DataFrame(dev_logging, columns=list(dev_logging.keys()))
    dev_logging_dataframe.to_csv(
        os.path.join(LOGGING_DIR, f'{timestring}-{dataset_name}-{model_name}-{mode.replace("train", "dev")}.csv'),
        header=True, index=False, sep='\t')
    if args_dataset.test_while_train:
        test_logging_dataframe = pandas.DataFrame(test_logging, columns=list(test_logging.keys()))
        test_logging_dataframe.to_csv(
            os.path.join(LOGGING_DIR, f'{timestring}-{dataset_name}-{model_name}-{mode.replace("train", "test")}.csv'),
            header=True, index=False, sep='\t')
    # Export configurations
    json.dump(kwargs, open(os.path.join(LOGGING_DIR, f'{timestring}-{dataset_name}-{model_name}-config.json'), 'w',
                           encoding='utf8'))
    terminate_logger(logger)
