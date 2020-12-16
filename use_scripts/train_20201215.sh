# 20201215 学習
# python3 preprocess_Hotels50k_mix.py
python3 train_box2mask.py --gpu_ids 2 \
                          --dataroot=datasets/Hotels50k_mix_mini_20201212/ \
                          --dataloader Hotels50k_mix \
                          --name Hotels50k_mix_box2mask_20201215 \
                          --use_gan --prob_bg 0.05 --label_nc 49  \
                          --output_nc 49 --model AE_maskgen_twostream \
                          --which_stream obj_context --tf_log --batchSize 8 \
                          --first_conv_stride 1 --first_conv_size 5  \
                          --conv_size 4 --num_layers 3 --use_resnetblock 1 \
                          --num_resnetblocks 1 --nThreads 2 --niter 200 \
                          --beta1 0.5 --objReconLoss bce --norm_layer instance \
                          --cond_in ctx_obj --gan_weight 0.1 \
                          --which_gan patch_multiscale --num_layers_D 3 \
                          --n_blocks 6 --fineSize 256 --use_output_gate \
                          --no_comb --contextMargin 3 --use_ganFeat_loss \
                          --min_box_size 32 --max_box_size 256 \
                          --add_dilated_layers --lr_control
python3 train_mask2image.py --gpu_ids 1 \
                            --dataroot=datasets/Hotels50k_mix_mini_20201212/ \
                            --dataloader Hotels50k_mix \
                            --name Hotels50k_mix_mask2image_20201215 \
                            --model pix2pixHD_condImg --no_instance \
                            --resize_or_crop select_region --loadSize 512 \
                            --fineSize 256 --contextMargin 3.0 --prob_bg 0.1 \
                            --label_nc 49 --output_nc 3 --load_image --tf_log \
                            --batchSize 8 --nThreads 2 --niter 150 \
                            --norm instance --n_downsample_global 4 \
                            --n_layers_D 3 --netG global_twostream \
                            --no_imgCond --min_box_size 64 \
                            --which_encoder ctx_label --use_skip \
                            --use_output_gate --mask_gan_input

