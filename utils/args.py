import argparse

def get_public_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--seed', type=int, default=2018)

    parser.add_argument('--bs', type=int, default=64)
    # seq_len denotes input history length, horizon denotes output future length
    parser.add_argument('--seq_len', type=int, default=12)
    parser.add_argument('--horizon', type=int, default=12)
    parser.add_argument('--input_dim', type=int, default=3)
    parser.add_argument('--output_dim', type=int, default=1)
#self_epoch_max
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--self_epoch_max', type=int, default=50)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--month', type=int, default=2)
    parser.add_argument('--num_month', type=int, default=2)
    return parser
