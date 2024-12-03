import torch


def get_optimizer(single_model, args):
    params_to_optimize = single_model.params_to_optimize()
    optimizer = torch.optim.AdamW(params_to_optimize,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  amsgrad=args.amsgrad
                                  )
    return optimizer