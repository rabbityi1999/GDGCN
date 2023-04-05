import torch
import numpy as np
import argparse
import time
import util
from engine import trainer
from pathlib import Path
import logging
import os
'''
This is the code for GDGCN
'''
parser = argparse.ArgumentParser()
parser.add_argument('--temporal_mode', type=str, default='temporal_gconv', help= 'temporal_gconv / tcn / attention / lstm / no_temporal/ mlp / mlp_new')

parser.add_argument('--layers', type=int, default=6, help='')
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--prefx', type=str, default='GDGCN',help='some words that remind you not to foget')
parser.add_argument('--model_name', type=str, default='gdgcn', help='gdgcn')
parser.add_argument('--data', type=str, default='./data/PEMS08', help='this is the input data path')
parser.add_argument('--patience', type=int, default=15, help='')
parser.add_argument('--epochs', type=int, default=200, help='')
parser.add_argument('--loss', type=str, default='huber', help='huber or mae')
parser.add_argument('--seq_length', type=int, default=12, help='default: 12')
parser.add_argument('--nhid', type=int, default=64, help='')
parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
parser.add_argument('--num_nodes', type=int, default=170, help='number of nodes')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--print_every', type=int, default=50, help='')
parser.add_argument('--seed', type=int, default=99, help='random seed')
parser.add_argument('--save', type=str, default='./garage/', help='save path')
parser.add_argument('--expid', type=int, default=1, help='experiment id')
parser.add_argument('--ablation_mode', type=str, default='none', help='none / no_multi-faceted_graph  / no_node_specific_temporal / no_time_specific_spatial / no_adaptive_patten_sharing')

args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
Path("log/").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler(f"log/{str(time.time())}_restore_{args.prefx}.log")
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

if not os.path.exists(args.save):
    os.makedirs(args.save)

def save_model(path: str, **save_dict):
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    torch.save(save_dict, path)

def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    engine = trainer(args, scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                     args.learning_rate, args.weight_decay, device)

    print("start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []
    count = 0
    mtrain_loss_list = []
    mval_loss_list = []

    for i in range(1, args.epochs + 1):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y, d) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx = trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            metrics = engine.train(trainx, trainy[:, 0, :, :], d)
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                logger.info(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]))

        t2 = time.time()
        train_time.append(t2 - t1)
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y, d) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:, 0, :, :], d)
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        logger.info(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)
        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        mtrain_loss_list.append(mtrain_loss)
        mval_loss_list.append(mvalid_loss)

        if len(his_loss) > 0 and mvalid_loss < np.min(his_loss):
            count = 0
        else:
            count += 1
            logger.info(f"no improve for {count} epochs")
        his_loss.append(mvalid_loss)
        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        logger.info(
            log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)))
        torch.save(engine.model.state_dict() , args.save + args.prefx + "_epoch_" + str(i) + "_" + str(round(mvalid_loss, 2)) + ".pth")
        if count >= args.patience:
            break
    logger.info("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    logger.info("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load( args.save + args.prefx + "_epoch_" + str(bestid + 1) + "_" + str(round(his_loss[bestid], 2)) + ".pth"))
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for iter, (x, y, d) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx, d).transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]
    logger.info("Training finished")
    log = 'The valid loss on best model is {:.4f}'
    logger.info(log.format(his_loss[bestid]))

    amae = []
    amape = []
    armse = []
    for i in range(12):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        metrics = util.metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        logger.info(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    logger.info(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))
    torch.save(engine.model.state_dict(), args.save + args.prefx + "_exp" + str(args.expid) + "_best_" + str(round(his_loss[bestid], 2)) + ".pth")
    logger.info(args)

if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    logger.info("Total time spent: {:.4f}".format(t2 - t1))
