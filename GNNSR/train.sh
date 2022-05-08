# train gnnsr-rec only with the reconstruction loss
python train.py --use_tb_logger --data_augmentation --max_iter 250 --loss_l1 --name train_gnnsr_rec --resume ./weights/train_gnnsr_rec/snapshot/net_20.pth --resume_optim ./weights/train_gnnsr_rec/snapshot1/optimizer_G_20.pth --resume_scheduler ./weights/train_gnnsr_rec/snapshot1/scheduler_20.pth


#train gnnsr with all losses, this model is based on the pretrained gnnsr-rec
#python train.py --use_tb_logger --max_iter 50 --loss_l1 --loss_adv --loss_perceptual --name train_gnnsr_gan --resume ./weights/train_gnnsr_rec/snapshot/net_best.pth --resume_optim ./weights/train_gnnsr_rec/snapshot/optimizer_G_best.pth --resume_scheduler ./weights/train_gnnsr_rec/snapshot/scheduler_best.pth

