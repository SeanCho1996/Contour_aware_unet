# %% import dependencies
from glob import glob
from torch.nn import DataParallel
from tensorboardX import SummaryWriter
from collections.abc import Sequence
from torchvision.transforms.transforms import Pad, Resize
import torchvision
from torch import nn
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
from model_with_border import BorderUNet
from dataset_with_border import *
from unet2d import UNet
from copy import deepcopy
import os
os.chdir('/home/zhaozixiao/projects/Torch_Border')
os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1, 2, 3"


# %% hyper parameters

batch_size = 8
epoch = 50
snapshot = 5
num_classes = 3

device = torch.device('cuda')

weight_name = 'model_512_unet2loss'


# %% import model
salt = BorderUNet(num_classes)

salt = DataParallel(salt)

salt.to(device)
# salt.load_state_dict(torch.load("/home/zhaozixiao/projects/Torch_UNet/model_voc_resunet/model_512_resunet_49.pth"))


scheduler_step = epoch // snapshot

# optimizer_ft = torch.optim.Adam(filter(lambda p: p.requires_grad, salt.parameters()), lr=1e-4)
optimizer_ft = torch.optim.SGD(
    filter(lambda p: p.requires_grad, salt.parameters()), lr=1e-3, momentum=0.9)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer_ft, step_size=13, gamma=0.1)

# %% pre-process


class ResizeSquarePad(Resize, Pad):
    def __init__(self, target_length, interpolation_strategy, pad_value):
        if not isinstance(target_length, (int, Sequence)):
            raise TypeError(
                "Size should be int or sequence. Got {}".format(type(target_length)))
        if isinstance(target_length, Sequence) and len(target_length) not in (1, 2):
            raise ValueError(
                "If size is a sequence, it should have 1 or 2 values")

        self.target_length = target_length
        self.interpolation_strategy = interpolation_strategy
        self.pad_value = pad_value
        Resize.__init__(self, size=(512, 512),
                        interpolation=self.interpolation_strategy)
        Pad.__init__(self, padding=(0, 0, 0, 0),
                     fill=self.pad_value, padding_mode="constant")

    def __call__(self, img):
        w, h = img.size
        if w > h:
            self.size = (
                int(np.round(self.target_length * (h / w))), self.target_length)
            img = Resize.__call__(self, img)

            total_pad = self.size[1] - self.size[0]
            half_pad = total_pad // 2
            self.padding = (0, half_pad, 0, total_pad - half_pad)
            return Pad.__call__(self, img)
        else:
            self.size = (self.target_length, int(
                np.round(self.target_length * (w / h))))
            img = Resize.__call__(self, img)

            total_pad = self.size[0] - self.size[1]
            half_pad = total_pad // 2
            self.padding = (half_pad, 0, total_pad - half_pad, 0)
            return Pad.__call__(self, img)


transform_img = torchvision.transforms.Compose([
    ResizeSquarePad(512, Image.BILINEAR, 255),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_mask = torchvision.transforms.Compose([
    ResizeSquarePad(512, Image.NEAREST, 255)
])

transform_border = torchvision.transforms.Compose([
    ResizeSquarePad(512, Image.NEAREST, 255)
])


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / \
                    (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss


class MulticlassDiceLoss(nn.Module):
    """
    requires input(prediction) dimension as [b, c, h, w]
    target(ground truth mask) dimension as [b, 1, h, w] where dimension 2 refers to the class index
    Can convert target to one_hot automatically and support ignore labels (should be in the form of list)
    """

    def __init__(self, ignore_labels=None):
        super(MulticlassDiceLoss, self).__init__()
        self.ignore_labels = ignore_labels

    def forward(self, input, target):

        num_ignore = 0 if self.ignore_labels == None else len(self.ignore_labels)

        n, _, h, w = target.shape[:]

        num_classes = input.shape[1]

        # initialize zeros for one_hot
        zeros = torch.zeros((n, (num_classes + num_ignore), h, w)).to(target.device)

        # decrease ignore labels' indexes into successive integers(eg: convert 0, 1, 2, 255 into 0, 1, 2, 3)
        for i in range(num_ignore):
            target[target == self.ignore_labels[i]] = num_classes + i

        # scatter to one_hot
        one_hot = zeros.scatter_(1, target, 1)

        dice = DiceLoss()
        totalLoss = 0

        # for indexes out of range, not compute corresponding loss
        for i in range(num_classes):
             diceLoss = dice(input[:, i], one_hot[:,i])
             totalLoss += diceLoss

        return totalLoss


class MeanIoU(nn.Module):
    """
    prediction: model's output with shape [b, c, h, w]
    mask: annotated mask with shape [b, h, w]
    ignore_labels: list of ignore_labels
    compute IoU of each class and the average
    """
    def __init__(self, ignore_labels=None):
        super(MeanIoU, self).__init__()
        self.ignore_labels = ignore_labels
    

    def forward(self, prediction, mask):
        predict = torch.argmax(prediction, dim=1)

        pure_mask = deepcopy(mask)
        pure_predict = deepcopy(predict)

        if self.ignore_labels != None:
            for ignore_label in self.ignore_labels:
                pure_mask = pure_mask.masked_select(mask.ne(ignore_label))
                pure_predict = pure_predict.masked_select(mask.ne(ignore_label))

        labels = torch.unique(pure_mask)
        iou_list = []

        for label in labels:
            sample_mask = pure_mask.eq(label)
            sample_predict = pure_predict.eq(label)
            intersection = (sample_mask * sample_predict).eq(True).sum().item()
            union = pure_mask.eq(label).sum().item() + pure_predict.eq(label).sum().item() - intersection
            iou_list.append(intersection / union)
        
        miou = sum(iou_list) / len(iou_list)
        return iou_list, miou


# %%
# Load data
img_dir = "./pets/"
# img_dir = "./polyp/"
# train_image_dir = "/home/zhaozixiao/projects/Torch_UNet/datasets/voc.devkit/train_full"
# val_image_dir = "/home/zhaozixiao/projects/Torch_UNet/datasets/voc.devkit/val_full"

# %%
X_train, y_train, X_val, y_val = ImageFetch(img_dir)
# X_train, y_train = trainImageFetch(train_image_dir)
# X_val, y_val = valImageFetch(val_image_dir)

# %%
train_data = SegDataset(X_train, y_train, 'train',
                        transform_img, transform_mask, transform_border)
val_data = SegDataset(X_val, y_val, 'val', transform_img,
                      transform_mask, transform_border)

a, b, c = train_data[0]
# %%
train_loader = DataLoader(train_data,
                          shuffle=RandomSampler(train_data),
                          batch_size=batch_size)

val_loader = DataLoader(val_data,
                        shuffle=False,
                        batch_size=batch_size)

# %%
criterion_ce = nn.CrossEntropyLoss(weight=torch.Tensor([1, 100]), ignore_index=255)
# criterion_ce = nn.CrossEntropyLoss(ignore_index=255)
criterion_dice = MulticlassDiceLoss(ignore_labels=[255])
# criterion_border = nn.BCEWithLogitsLoss()

miou_accuracy = MeanIoU(ignore_labels=[255])

# %%

def train(train_loader, model):
    running_loss = 0.0
    acc = 0.0
    data_size = len(train_data)

    model.train()

    for inputs, masks, borders in tqdm(train_loader):
        inputs, masks, borders = inputs.to(device), masks.long().to(
            device), borders.long().to(device)
        optimizer_ft.zero_grad()

        # weight = torch.Tensor([1, int(len(borders[borders == 0])/len(borders[borders == 1]))]).to(device)

        out, init_border = model(inputs)

        # predict = torch.argmax(nn.Softmax(dim=1)(out), dim=1)
        # pure_mask = masks.masked_select(masks.ne(255))
        # pure_predict = predict.masked_select(masks.ne(255))
        # acc += pure_mask.cpu().eq(pure_predict.cpu()).sum().item()/len(pure_mask)

        acc += miou_accuracy(out, masks)[1]
        # print(miou)

        # loss_init = criterion_seg(init_seg, masks)
        criterion_ce.to(device)
        loss_border = criterion_ce(init_border, borders)
        # loss_seg = criterion_dice(init_seg, masks.unsqueeze(1))
        loss_final = criterion_dice(out, masks.unsqueeze(1))
        # loss_seg = criterion_seg(out, masks)

        # loss = loss_init + loss_border + loss_seg
        loss = loss_border + loss_final

        loss.backward()
        optimizer_ft.step()
        # print(loss.item())
        running_loss += loss.item() * batch_size

    epoch_loss = running_loss / data_size
    accuracy = acc / len(train_loader)
    return epoch_loss, accuracy


def test(test_loader, model):
    running_loss = 0.0
    acc = 0.0
    data_size = len(val_data)

    model.eval()

    with torch.no_grad():
        for inputs, masks, borders in test_loader:
            inputs, masks, borders = inputs.to(device), masks.long().to(
                device), borders.long().to(device)

            out, fine_border = model(inputs)

            # _, miou = miou_accuracy(out, masks)

            # predict = torch.argmax(nn.Softmax(dim=1)(out), dim=1)
            # pure_mask = masks.masked_select(masks.ne(255))
            # pure_predict = predict.masked_select(masks.ne(255))
            # acc += pure_mask.cpu().eq(pure_predict.cpu()).sum().item()/len(pure_mask)
            acc += miou_accuracy(out, masks)[1]


            # loss_init = criterion_seg(init_seg, masks)
            loss_border = criterion_ce(fine_border, borders)
            # loss_seg = criterion_dice(init_seg, masks.unsqueeze(1))
            loss_final = criterion_dice(out, masks.unsqueeze(1))

            # loss = loss_seg + loss_init + loss_border
            loss = loss_border + loss_final

            running_loss += loss.item() * batch_size

    epoch_loss = running_loss / data_size
    accuracy = acc / len(test_loader)
    return epoch_loss, accuracy  # precision


# %%
num_snapshot = 0
best_acc = 0
writer = SummaryWriter("./log/polyp_borderloss")


for epoch_ in range(epoch):
    train_loss, train_acc = train(train_loader, salt)
    val_loss, accuracy = test(val_loader, salt)
    exp_lr_scheduler.step()

    writer.add_scalar('loss/train', train_loss, epoch_)
    writer.add_scalar('loss/valid', val_loss, epoch_)
    writer.add_scalar('accuracy', accuracy, epoch_)
    # writer.add_scalars('Val_loss', {'val_loss': val_loss}, n_iter)

    if accuracy > best_acc:
      best_acc = accuracy
      best_param = salt.state_dict()

    print('epoch: {} train_loss: {:.3f} train_accuracy: {:.3f} val_loss: {:.3f} val_accuracy: {:.3f}'.format(
        epoch_ + 1, train_loss, train_acc, val_loss, accuracy))
    torch.save(salt.module.state_dict(),
               './model_pet_borderloss_512/' + weight_name + '_%d.pth' % epoch_)
writer.close()

# %%
# salt = BorderUNet(2)
# salt.load_state_dict(torch.load("/home/zhaozixiao/projects/Torch_Border/model_pet_borderfusion_512/model_512_borderunet_49.pth"))
# salt.cuda()

# for i, img in enumerate(X_val):
#     ori_image = img
#     name = os.path.splitext(ori_image.filename.split("/")[-1])[0]
#     img = transform_img(img)

#     img = img.cuda()

#     out, init_seg, border = salt(img.unsqueeze(0))

#     predict = out.squeeze(0)
#     predict = nn.Softmax(dim=0)(predict)
#     predict = torch.argmax(predict, dim=0)

#     border = border.squeeze(0)
#     border = nn.Softmax(dim=0)(border)
#     border = torch.argmax(border, dim=0)

#     w, h = ori_image.size
#     if w > h:
#         re_h = int(np.round(512 * (h / w)))
#         total_pad = 512 - re_h
#         half_pad = total_pad // 2
#         out = predict[half_pad: half_pad + re_h, :]
#         b = border[half_pad: half_pad + re_h, :]
#     else:
#         re_w = int(np.round(512 * (w / h)))
#         total_pad = 512 - re_w
#         half_pad = total_pad // 2
#         out = predict[:, half_pad: half_pad + re_w]
#         b = border[:, half_pad: half_pad + re_w]

#     predict = cv2.resize(out.cpu().numpy(), (w, h),
#                          interpolation=cv2.INTER_NEAREST)
#     border_pred = cv2.resize(b.cpu().numpy(), (w, h),
#                          interpolation=cv2.INTER_NEAREST)

#     # out_img = np.zeros((320, 320, 3))

#     out_png = Image.fromarray(predict.astype(np.uint8))

#     border_pred[border_pred == 1] = 255
#     out_border = Image.fromarray(border_pred.astype(np.uint8))

#     palette = []
#     for j in range(256):
#         palette.extend((j, j, j))
#     palette[:3*21] = np.array([[0, 0, 0],
#         [128, 0, 0],
#         [0, 128, 0],
#         [128, 128, 0],
#         [0, 0, 128],
#         [128, 0, 128],
#         [0, 128, 128],
#         [128, 128, 128],
#         [64, 0, 0],
#         [192, 0, 0],
#         [64, 128, 0],
#         [192, 128, 0],
#         [64, 0, 128],
#         [192, 0, 128],
#         [64, 128, 128],
#         [192, 128, 128],
#         [0, 64, 0],
#         [128, 64, 0],
#         [0, 192, 0],
#         [128, 192, 0],
#         [0, 64, 128]
#         ], dtype='uint8').flatten()
#     out_png.putpalette(palette)
#     # out_border.putpalette(palette)

#     out_png.save(
#         "/home/zhaozixiao/projects/Torch_Border/polyp_results/" + name + ".png")
#     out_border.save(
#         "/home/zhaozixiao/projects/Torch_Border/polyp_results/border_" + name + ".png")

# %%
