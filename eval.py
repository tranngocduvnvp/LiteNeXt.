import torch
import torchmetrics
import numpy as np
import time
import torchmetrics.functional.classification as Fmstric

@torch.no_grad()
def validation(model, device, test_loader, epoch, perf_measure, phase):
    t = time.time()
    model.eval()
    perf_accumulator = []
    mIOU = []
    Precision = []
    Recall = []
    F1_score = []
    for batch_idx, (data, target, idx) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        if torch.sum(target) == 0:
            continue
        output = model(data)
        perf_accumulator.append(perf_measure(output, target).item())
        mIOU.append(Fmstric.binary_jaccard_index(torch.sigmoid(output), target>0.5).item())
        Precision.append(torchmetrics.functional.classification.binary_precision(torch.sigmoid(output), target>0.5).item())
        Recall.append(torchmetrics.functional.classification.binary_recall(torch.sigmoid(output), target>0.5).item())
        F1_score.append(torchmetrics.functional.classification.binary_f1_score(torch.sigmoid(output), target>0.5).item())

        if batch_idx + 1 < len(test_loader):
            print(
                "\r{}  Epoch: {} [{}/{} ({:.1f}%)]\tDice: {:.6f}\tmIOU: {:.6f}\tPrecision: {:.6f}\tRecall: {:.6f}\tF1_score: {:.6f}\tTime: {:.6f}".format(
                    phase, epoch, batch_idx + 1, len(test_loader), 100.0 * (batch_idx + 1) / len(test_loader),
                    np.mean(perf_accumulator), np.mean(mIOU), np.mean(Precision), np.mean(Recall), np.mean(F1_score), time.time() - t, ), end="", )
        else:
            print(
                "\r{}  Epoch: {} [{}/{} ({:.1f}%)]\tDice: {:.6f}\tmIOU: {:.6f}\tPrecision: {:.6f}\tRecall: {:.6f}\tF1_score: {:.6f}\tTime: {:.6f}".format(
                    phase, epoch, batch_idx + 1, len(test_loader), 100.0 * (batch_idx + 1) / len(test_loader),
                    np.mean(perf_accumulator), np.mean(mIOU), np.mean(Precision), np.mean(Recall), np.mean(F1_score), time.time() - t, ))

    return np.mean(perf_accumulator), np.std(perf_accumulator)

@torch.no_grad()
def test(model, device, test_loader, epoch, perf_measure, phase):
    t = time.time()
    model.eval()

    mDice_NM = []
    mIOU_NM = []    
    Precision_NM = []
    Recall_NM = []

    mDice_ANM = []
    mIOU_ANM = []    
    Precision_ANM = []
    Recall_ANM = []

    for batch_idx, (data, target, idx) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        if torch.sum(target) == 0:
            continue
        output = model(data)

        if idx==0:
            mDice_NM.append(perf_measure(output, target).item())
            mIOU_NM.append(Fmstric.binary_jaccard_index(torch.sigmoid(output), target>0.5).item())
            Precision_NM.append(torchmetrics.functional.classification.binary_precision(torch.sigmoid(output), target>0.5).item())
            Recall_NM.append(torchmetrics.functional.classification.binary_recall(torch.sigmoid(output), target>0.5).item())
        else:
            mDice_ANM.append(perf_measure(output, target).item())
            mIOU_ANM.append(Fmstric.binary_jaccard_index(torch.sigmoid(output), target>0.5).item())
            Precision_ANM.append(torchmetrics.functional.classification.binary_precision(torch.sigmoid(output), target>0.5).item())
            Recall_ANM.append(torchmetrics.functional.classification.binary_recall(torch.sigmoid(output), target>0.5).item())

    print("\033[1m" + "========== Test ==============" + "\033[0m")
    print("\033[44m\033[37mNormal\033[0m")
    print(
        "\r{}  Epoch: {} [{}/{} ({:.1f}%)]\tDice: {:.6f}\tmIOU: {:.6f}\tPrecision: {:.6f}\tRecall: {:.6f}\tTime: {:.6f}".format(
            phase, epoch, batch_idx + 1, len(test_loader), 100.0 * (batch_idx + 1) / len(test_loader),
            np.mean(mDice_NM), np.mean(mIOU_NM), np.mean(Precision_NM), np.mean(Recall_NM), time.time() - t, ))

    print("\033[43m\033[30mAbNormal\033[0m")
    print(
        "\r{}  Epoch: {} [{}/{} ({:.1f}%)]\tDice: {:.6f}\tmIOU: {:.6f}\tPrecision: {:.6f}\tRecall: {:.6f}\tTime: {:.6f}".format(
            phase, epoch, batch_idx + 1, len(test_loader), 100.0 * (batch_idx + 1) / len(test_loader),
            np.mean(mDice_ANM), np.mean(mIOU_ANM), np.mean(Precision_ANM), np.mean(Recall_ANM), time.time() - t, ))
    print("\033[42m\033[31mAverage\033[0m")
    print(
        "\r{}  Epoch: {} [{}/{} ({:.1f}%)]\tDice: {:.6f}\tmIOU: {:.6f}\tPrecision: {:.6f}\tRecall: {:.6f}\tTime: {:.6f}".format(
            phase, epoch, batch_idx + 1, len(test_loader), 100.0 * (batch_idx + 1) / len(test_loader),
            np.mean(mDice_NM+mDice_ANM), np.mean(mIOU_NM + mIOU_ANM), np.mean(Precision_NM + Precision_ANM), np.mean(Recall_NM + Recall_ANM), time.time() - t, ))

    return np.mean(mDice_NM+mDice_ANM), np.std(mDice_NM+mDice_ANM)