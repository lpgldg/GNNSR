from models.archs import common
import torch
import torch.nn as nn
import torch.nn.functional as F

class Pixel_Unshuffle(nn.Module):
    def __init__(self,):
        super(Pixel_Unshuffle,self).__init__()

    def forward(self, put, down_scale):
        c = put.shape[1]
        kernel = torch.zeros(size = [down_scale*down_scale*c,1, down_scale,down_scale], device=put.device)
        for y in range(down_scale):
            for x in range(down_scale):
                kernel[x + y*down_scale::down_scale*down_scale,0,y,x]=1
        return F.conv2d(put, kernel,stride = down_scale,groups=c)


class GAT(nn.Module):
    def __init__(self,):
        super(GAT, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, h, adj):
        energy = torch.bmm(h, h.permute(0, 2, 1))
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        energy_new = torch.mul(energy_new, adj)
        attention = self.softmax(energy_new)
        return attention




class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [
            common.RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size,1,1))
        self.body = nn.Sequential(*modules_body)
        ##GAT
        self.gat = GAT()
        modules_tail = [conv(n_feat, n_feat, kernel_size, 1, 1)]
        self.tail = nn.Sequential(*modules_tail)
        self.LTE = common.LTE(requires_grad=True)
        modules_scale = [conv(n_feat, 4, kernel_size,1,1)]
        self.scale = nn.Sequential(*modules_scale)
        self.down = nn.Upsample(scale_factor=0.25, mode='bicubic', align_corners=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.act = nn.ReLU(inplace=True)
   
        self.down = Pixel_Unshuffle()

        self.last = nn.Conv2d(n_feat *2, n_feat, 3, 1, 1)
        self.last2 = nn.Conv2d(n_feat *2, n_feat, 3, 1, 1)
        self.up = nn.ConvTranspose2d(n_feat, n_feat, kernel_size=8, stride=4, padding=2, dilation=1, bias=True)
        self.down2 = nn.Conv2d(n_feat, n_feat, kernel_size=8, stride=4, padding=2, dilation=1, bias=True)
        self.compress = nn.Conv2d(n_feat *2, n_feat,kernel_size=1,stride=1, padding=0, dilation=1,bias=True)

    def forward(self, x, y, y_down):
        
        # RCAB LR(res)-->feature
        res = self.body(x)  #9 64 40 40
        ref = self.body(y)  
        refl = self.body(y_down) 

        #lr_up--->shape
        res_up = self.up(res)
        ones = torch.ones_like(res_up)
        ones = F.unfold(ones,kernel_size=16,padding=0,stride=4)
        
        #ref_HR-->patches
        ref_up_patches = F.unfold(ref, kernel_size=16,padding=0,stride=4).permute(0,2,1)
        
      
       

        ref1 = ref + y
        ref_down = refl + y_down   
        _,_,h,w =res.size()

        # reduce ref fetures
        ref = self.scale(ref)
        ref =self.down(ref,4)

        # lr-->patch BxN1xf
        lr_patches = F.unfold(res,kernel_size=4,padding=0, stride=1).permute(0,2,1)
        one = torch.ones_like(res)
        one = F.unfold(one,kernel_size=4, padding=0, stride=1)
        
        #refl-->patch BxfxN2
        refl_patches = F.unfold(refl,kernel_size=4,padding=0,stride=1) 
        #ref-->patch BxN2xf
        
        ref_patches =F.unfold(ref,kernel_size=4,padding=0,stride=1).permute(0,2,1)
   
    
        lr = F.normalize(lr_patches, dim=2)
        refl = F.normalize(refl_patches, dim=1)
        ## BxN1xN2 
        corr = torch.bmm(lr, refl)
   
        zero = torch.zeros_like(corr)
        score,_ = torch.topk(corr, k=3, dim=-1, largest=True, sorted=True)

        score = score[:,:,-1:]

        score = score.repeat(1,1,corr.size()[-1]) #BxN1xN2

        score = torch.where(corr>=score, corr, zero)  #BxN1xN2

        out = torch.bmm(score, ref_patches) #BxN1xf
        out_up = torch.bmm(score,ref_up_patches)

        out_up = F.fold(out_up.permute(0,2,1),kernel_size=16,stride=4,output_size=(4*h,4*w))
        ones = F.fold(ones,kernel_size=16,stride=4,output_size=(4*h,4*w))
        out_up = out_up/ones
        out_up = torch.cat([res_up, out_up],1)
        out_up = self.compress(out_up)
        out_up = self.down2(out_up)
        
      
        out = F.fold(out.permute(0,2,1),kernel_size=4, stride=1, output_size=(h,w))
        one = F.fold(one,kernel_size=4, stride=1, output_size=(h,w))
        out = out/one
        out = torch.cat([res,out], 1)
        out = self.last(out)
        
        out = self.act(self.last2(torch.cat([out,out_up],1)))
        out = out + x
        
        
        return out, ref1, ref_down

class GNNSR(nn.Module):
    def __init__(self, args,conv=nn.Conv2d):
        super(GNNSR, self).__init__()
        n_resgroups = 5
        n_resblocks = 5
        n_feats = 64
        kernel_size = 3
        reduction = 16
        scale = 4
        act = nn.ReLU(True)
      
        # define head module
        modules_head = [conv(3, n_feats, kernel_size, 1, 1)]
        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=1, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size, 1, 1))

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, 3, kernel_size, 1, 1)]

        #self.add_mean = common.MeanShift(255, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.la = common.LAM_Module(n_feats)
        self.last_conv = nn.Conv2d(n_feats *(n_resgroups+1), n_feats, 3, 1, 1)
        self.last = nn.Conv2d(n_feats *2, n_feats, 3, 1, 1)
        self.tail = nn.Sequential(*modules_tail)
        self.last2_conv = nn.Conv2d(30, 3, 3, 1, 1)
        self.last2 = nn.Conv2d(6, 3, 3, 1, 1)

    def forward(self, x, y, y_down):

        lrsr = F.interpolate(x, scale_factor=4, mode='bicubic')

        x = self.head(x)
        y = self.head(y)
        y_down = self.head(y_down)

        res = x
        for name, midlayer in self.body._modules.items():
            if name=='5':
                res =res
            else:
                res, y, y_down = midlayer(res, y, y_down)
            if name=='0':
                res1 = res.unsqueeze(1)
            else:
                res1 = torch.cat([res.unsqueeze(1) ,res1] ,1)

        out1 = res
        res = self.la(res1)  
        out2 = self.last_conv(res)  
        out = torch.cat([out1 ,out2] ,1 )  
        res = self.last(out)


        x = self.tail(res)  
      
        return x+lrsr

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
