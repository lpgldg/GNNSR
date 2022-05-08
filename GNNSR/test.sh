# test the gnnsr_rec model (trained with only reconstruction loss)
python test.py --resume './pretrained_weights/gnnsr_rec.pth' --testset TestSet --name gnnsr_rec_TestSet --ref_level 1 # test on CUFED5 testing set
# python test.py --resume './pretrained_weights/gnnsr_rec.pth' --testset Sun80 --name gnnsr_rec_Sun80  # test on Sun80 dataset.
# python test.py --resume './pretrained_weights/gnnsr_rec.pth' --testset Urban100 --name gnnsr_rec_Urban100  # test on Urban100 dataset.


# # test the gnnsr model (trained with all losses)
# python test.py --resume './pretrained_weights/gnnsr.pth' --testset TestSet --name gnnsr_TestSet --ref_level 1  # test on CUFED5 testing set, using single ref.
# python test.py --resume './pretrained_weights/gnnsr.pth' --testset Sun80 --name gnnsr_Sun80  # test on Sun80 dataset.
# python test.py --resume './pretrained_weights/gnnsr.pth' --testset Urban100 --name gnnsr_Urban100  # test on Urban100 dataset.
