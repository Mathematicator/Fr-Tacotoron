from torch import nn


class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        print ("loss mel out : {} , loss mel postnet : {} , gate loss : {} ".format(nn.MSELoss()(mel_out, mel_target), nn.MSELoss()(mel_out_postnet, mel_target), nn.BCEWithLogitsLoss()(gate_out, gate_target)))
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        file = open('mel_loss_train.txt', "a+", encoding="utf-8")
        file.write(str(mel_loss.float().data.cpu().numpy())+"\n")
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        return mel_loss + gate_loss
