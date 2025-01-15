if args.dataset == 'modelnet40':
        train_loader = DataLoader(
            ModelNet40(num_points=args.num_points, partition='train',
                       gaussian_noise=args.gaussian_noise, unseen=args.unseen,
                       factor=args.factor),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True
        )
        test_loader = DataLoader(
            ModelNet40(num_points=args.num_points, partition='test',
                       gaussian_noise=args.gaussian_noise, unseen=args.unseen,
                       factor=args.factor),
            batch_size=args.test_batch_size,
            shuffle=False,
            drop_last=False
        )
    elif args.dataset == 'custom':
        # ???????? CustomDataset ???????????????? partition='train' ???? partition='test'
        train_loader = DataLoader(
            CustomDataset(num_points=args.num_points,
                          root=args.root,
                          partition='train',
                          gaussian_noise=args.gaussian_noise,
                          factor=args.factor),
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True
        )
        test_loader = DataLoader(
            CustomDataset(num_points=args.num_points,
                          root=args.root,
                          partition='test',
                          gaussian_noise=args.gaussian_noise,
                          factor=args.factor),
            batch_size=args.test_batch_size,
            shuffle=False,
            drop_last=False
        )
    else:
        raise Exception("not implemented")
