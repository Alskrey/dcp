def main():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--root', required=True, help='the data path')
    parser.add_argument('--model', type=str, default='dcp', metavar='N',
                        choices=['dcp'],
                        help='Model to use, [dcp]')
    parser.add_argument('--emb_nn', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Embedding nn to use, [pointnet, dgcnn]')
    parser.add_argument('--pointer', type=str, default='transformer', metavar='N',
                        choices=['identity', 'transformer'],
                        help='Attention-based pointer generator to use, [identity, transformer]')
    parser.add_argument('--head', type=str, default='svd', metavar='N',
                        choices=['mlp', 'svd', ],
                        help='Head to use, [mlp, svd]')
    parser.add_argument('--emb_dims', type=int, default=512, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--n_blocks', type=int, default=1, metavar='N',
                        help='Num of blocks of encoder&decoder')
    parser.add_argument('--n_heads', type=int, default=4, metavar='N',
                        help='Num of heads in multiheadedattention')
    parser.add_argument('--ff_dims', type=int, default=1024, metavar='N',
                        help='Num of dimensions of fc in transformer')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='N',
                        help='Dropout ratio in transformer')
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=600, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', action='store_true', default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', action='store_true', default=True,
                        help='evaluate the model')
    parser.add_argument('--cycle', type=bool, default=False, metavar='N',
                        help='Whether to use cycle consistency')
    parser.add_argument('--gaussian_noise', type=bool, default=False, metavar='N',
                        help='Wheter to add gaussian noise')
    parser.add_argument('--unseen', type=bool, default=False, metavar='N',
                        help='Wheter to test on unseen category')
    parser.add_argument('--num_points', type=int, default=1024, metavar='N',
                        help='Num of points to use')
    parser.add_argument('--dataset', type=str, default='custom', choices=['modelnet40','custom'], metavar='N',
                        help='dataset to use')
    parser.add_argument('--factor', type=float, default=4, metavar='N',
                        help='Divided factor for rotations')
    parser.add_argument('--model_path', type=str, default='./pretrained/dcp_v2.t7', metavar='N',
                        help='Pretrained model path')








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
