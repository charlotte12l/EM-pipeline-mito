import torch
import torchvision.utils as vutils
import torch.nn.functional as F

N = 8 # default maximum number of sections to show

def prepare_data(volume, label, output, aux =False):
    if aux ==1:
        aux_output = output[0]
        output = output[-1]
        aux_label = label.clone().float()
        aux_label = F.interpolate(aux_label, size=(aux_output.size(2),aux_output.size(3), aux_output.size(4)), mode='trilinear', align_corners=True)
    if len(volume.size()) == 4:   # 2D Inputs
        if volume.size()[0] > N:
            if aux ==1:
                return volume[:N], label[:N], output[:N], aux_output[:N], aux_label[:N]
            else:
                return volume[:N], label[:N], output[:N]
        else:
            if aux ==1:
                return volume, label, output, aux_output, aux_label
            return volume, label, output
    elif len(volume.size()) == 5: # 3D Inputs
        if aux ==1:
            aux_output, aux_label = aux_output[0].permute(1,0,2,3), aux_label[0].permute(1,0,2,3)
        volume, label, output = volume[0].permute(1,0,2,3), label[0].permute(1,0,2,3), output[0].permute(1,0,2,3)
        if volume.size()[0] > N:
            if aux ==1:
                return volume[:N], label[:N], output[:N], aux_output[:N], aux_label[:N]
            else:
                return volume[:N], label[:N], output[:N]
        else:
            if aux ==1:
                return volume, label, output, aux_output, aux_label
            return volume, label, output

def visualize(volume, label, output, iteration, writer, aux = False):
    if aux ==1:
        volume, label, output, aux_output, aux_label = prepare_data(volume, label, output, aux=True)
    else:
        volume, label, output = prepare_data(volume, label, output, aux=False)

    sz = volume.size() # z,c,y,x
    volume_visual = volume.detach().cpu().expand(sz[0],3,sz[2],sz[3])
    output_visual = output.detach().cpu().expand(sz[0],3,sz[2],sz[3])
    label_visual = label.detach().cpu().expand(sz[0],3,sz[2],sz[3])
    

    volume_show = vutils.make_grid(volume_visual, nrow=8, normalize=True, scale_each=True)
    output_show = vutils.make_grid(output_visual, nrow=8, normalize=True, scale_each=True)
    label_show = vutils.make_grid(label_visual, nrow=8, normalize=True, scale_each=True)

    output_visual[output_visual<0] = -1
    output2_show = vutils.make_grid(output_visual, nrow=8, normalize=True, scale_each=True)
    
    writer.add_image('Input', volume_show, iteration)
    writer.add_image('Label', label_show, iteration)
    writer.add_image('Output', output_show, iteration)
    writer.add_image('Output2', output2_show, iteration)

    if aux ==1:
        sz_aux = aux_output.size()
        aux_output_visual = aux_output.detach().cpu().expand(sz_aux[0],3,sz_aux[2],sz_aux[3])
        aux_label_visual = aux_label.detach().cpu().expand(sz_aux[0],3,sz_aux[2],sz_aux[3])

        aux_output_show = vutils.make_grid(aux_output_visual, nrow=8, normalize=True, scale_each=True)
        aux_label_show = vutils.make_grid(aux_label_visual, nrow=8, normalize=True, scale_each=True)

        writer.add_image('Aux Label', aux_label_show, iteration)
        writer.add_image('Aux Output', aux_output_show, iteration)


def visualize_aff(volume, label, output, iteration, writer):
    volume, label, output = prepare_data(volume, label, output)

    sz = volume.size() # z,c,y,x
    canvas = []
    volume_visual = volume.detach().cpu().expand(sz[0],3,sz[2],sz[3])
    canvas.append(volume_visual)
    output_visual = [output[:,i].detach().cpu().unsqueeze(1).expand(sz[0],3,sz[2],sz[3]) for i in range(3)]
    label_visual = [label[:,i].detach().cpu().unsqueeze(1).expand(sz[0],3,sz[2],sz[3]) for i in range(3)]
    canvas = canvas + output_visual
    canvas = canvas + label_visual
    canvas_merge = torch.cat(canvas, 0)
    canvas_show = vutils.make_grid(canvas_merge, nrow=8, normalize=True, scale_each=True)

    writer.add_image('Affinity', canvas_show, iteration)