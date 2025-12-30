import torch
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs


def launch_training_task(
    dataset,
    model,
    model_logger,
    learning_rate = 1e-5,
    weight_decay = 1e-2,
    num_workers = 8,
    save_steps = None,
    num_epochs = 1,
    gradient_accumulation_steps = 1,
    find_unused_parameters = False,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        gradient_accumulation_steps = args.gradient_accumulation_steps
        find_unused_parameters = args.find_unused_parameters
    
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)

    accelerator = Accelerator(
        gradient_accumulation_steps = gradient_accumulation_steps,
        kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)],
        log_with = "tensorboard",
        project_dir = "logs"
    )
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    accelerator.init_trackers("V1")

    global_step = 0
    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                loss = model(data)

                if global_step % 1 == 0:
                    accelerator.log(
                        {
                            "train/loss": loss.item(),
                            "epoch": epoch_id
                        },
                        step = global_step
                    )    
                global_step += 1

                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps)
                scheduler.step()
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)