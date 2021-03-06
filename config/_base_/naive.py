batch_size = 256
num_epochs = 100

cl_strategy = dict(
    type='Naive',
    train_mb_size=batch_size,
    eval_mb_size=batch_size,
    train_epochs=num_epochs,
    optimizer=dict(
        type='SGD',
        lr=1.0,
        weight_decay=0.0,
        momentum=0.9),
    scheduler=dict(
        type='StepLR',
        step_size=60,
        gamma=0.1),
    loss=dict(
        type='CrossEntropyLoss',
        loss_weight=1.0))
